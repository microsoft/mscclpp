#!/usr/bin/env python3
"""Standalone SGLang serving benchmark script.

Usage examples:

  # Benchmark with a JSONL dataset (MAI-style prompts)
  python bench_sglang.py \\
    --dataset ~/prompts/single_turn_completions_with_si_fixed.jsonl \\
    --tokenizer /tmp/sgl_mai3_5b \\
    --host 127.0.0.1 --port 30000 \\
    --max-concurrency 64 --num-prompts 1729

  # Benchmark a Qwen model via OpenAI-compatible API
  python bench_sglang.py \\
    --backend openai \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --dataset ~/prompts/my_prompts.jsonl \\
    --tokenizer Qwen/Qwen2.5-7B-Instruct \\
    --host 127.0.0.1 --port 30000

  # Synthetic random prompts (no dataset file needed)
  python bench_sglang.py \\
    --backend openai \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --tokenizer Qwen/Qwen2.5-7B-Instruct \\
    --random-input-len 512 --random-output-len 256 --num-prompts 100
"""

import argparse
import asyncio
import json
import random
import resource
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DatasetRow:
    prompt: str | list[int]
    prompt_len: int
    output_len: int


@dataclass
class RequestInput:
    prompt: str | list[int]
    api_url: str
    prompt_len: int
    output_len: int
    model: str


@dataclass
class RequestOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    prompt_len: int = 0
    output_len: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
_READ_BUFSIZE = 10 * 1024 ** 2   # 10 MB


def _create_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS),
        read_bufsize=_READ_BUFSIZE,
    )


def _remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


# ---------------------------------------------------------------------------
# Backend: SGLang native  (POST /generate, SSE streaming)
# ---------------------------------------------------------------------------

async def _request_sglang(
    inp: RequestInput,
    pbar: tqdm | None = None,
) -> RequestOutput:
    """Send a single request to the SGLang /generate endpoint with streaming."""
    async with _create_session() as session:
        payload: dict[str, Any] = {
            ("text" if isinstance(inp.prompt, str) else "input_ids"): inp.prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": inp.output_len,
                "ignore_eos": True,
            },
            "stream": True,
        }

        output = RequestOutput(prompt_len=inp.prompt_len)
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        last_output_len = 0
        generated_text = ""
        output_len = inp.output_len
        latency = 0.0

        try:
            async with session.post(url=inp.api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = _remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            continue

                        data = json.loads(chunk)
                        if data.get("text"):
                            timestamp = time.perf_counter()
                            generated_text = data["text"]
                            meta_info = data["meta_info"]
                            output_len = meta_info["completion_tokens"]

                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft
                            else:
                                num_new_tokens = output_len - last_output_len
                                if num_new_tokens > 0:
                                    chunk_gap = timestamp - most_recent_timestamp
                                    adjust_itl = chunk_gap / num_new_tokens
                                    output.itl.extend([adjust_itl] * num_new_tokens)

                            most_recent_timestamp = timestamp
                            last_output_len = output_len

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = output_len
                else:
                    error_body = await response.text()
                    output.error = f"{response.reason or ''}: {error_body}"
        except Exception:
            output.error = "".join(traceback.format_exception(*sys.exc_info()))

    if pbar:
        pbar.update(1)
    return output


# ---------------------------------------------------------------------------
# Backend: OpenAI-compatible  (POST /v1/completions, SSE streaming)
# Works for Qwen, Llama, etc. served via SGLang with --served-model-name
# ---------------------------------------------------------------------------

async def _request_openai(
    inp: RequestInput,
    pbar: tqdm | None = None,
) -> RequestOutput:
    """Send a single request to /v1/completions with streaming."""
    async with _create_session() as session:
        payload: dict[str, Any] = {
            "model": inp.model,
            "prompt": inp.prompt,
            "max_tokens": inp.output_len,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
        }

        output = RequestOutput(prompt_len=inp.prompt_len)
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        generated_text = ""
        output_len = inp.output_len
        latency = 0.0

        try:
            async with session.post(url=inp.api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = _remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            continue

                        data = json.loads(chunk)
                        text_piece = data["choices"][0].get("text", "")
                        if text_piece:
                            timestamp = time.perf_counter()

                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)

                            most_recent_timestamp = timestamp
                            generated_text += text_piece

                            usage = data.get("usage") or {}
                            output_len = usage.get("completion_tokens", output_len)

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = output_len
                else:
                    error_body = await response.text()
                    output.error = f"{response.reason or ''}: {error_body}"
        except Exception:
            output.error = "".join(traceback.format_exception(*sys.exc_info()))

    if pbar:
        pbar.update(1)
    return output


BACKEND_FUNCS = {
    "sglang": _request_sglang,
    "openai": _request_openai,
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl_dataset(path: str, num_prompts: int) -> list[DatasetRow]:
    """Load prompts from a JSONL file. Each line needs: prompt, prompt_len, output_len."""
    rows: list[DatasetRow] = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            rows.append(DatasetRow(
                prompt=data["prompt"],
                prompt_len=data["prompt_len"],
                output_len=data["output_len"],
            ))
    if not rows:
        raise ValueError(f"No requests found in {path}")
    if len(rows) < num_prompts:
        raise ValueError(
            f"File has {len(rows)} prompts, but --num-prompts is {num_prompts}"
        )
    return rows[:num_prompts]


def generate_random_dataset(
    tokenizer: AutoTokenizer,
    num_prompts: int,
    input_len: int,
    output_len: int,
) -> list[DatasetRow]:
    """Generate synthetic prompts with random token ids."""
    vocab_size = tokenizer.vocab_size
    rows: list[DatasetRow] = []
    for _ in range(num_prompts):
        token_ids = [random.randint(0, vocab_size - 1) for _ in range(input_len)]
        prompt_text = tokenizer.decode(token_ids)
        rows.append(DatasetRow(prompt=prompt_text, prompt_len=input_len, output_len=output_len))
    return rows


# ---------------------------------------------------------------------------
# Request rate generator
# ---------------------------------------------------------------------------

async def _get_requests(
    input_requests: list[DatasetRow],
    request_rate: float,
):
    for request in input_requests:
        yield request
        if request_rate != float("inf"):
            interval = 1.0 / request_rate
            await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    total_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p50_ttft_ms: float
    std_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p50_itl_ms: float
    std_itl_ms: float
    p90_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    max_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p90_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float
    # Aggregate prefill/decode throughput (total tokens / total phase time)
    agg_prefill_throughput: float   # total_input_tokens / sum(ttfts)
    agg_decode_throughput: float    # total_decode_tokens / sum(decode_times)
    # Per-request prefill throughput (input tok/s) — P10 = slow tail
    mean_prefill_throughput: float
    median_prefill_throughput: float
    p10_prefill_throughput: float
    p25_prefill_throughput: float
    # Per-request decode time (ms) — P90/P99 = slow tail
    mean_decode_ms: float
    median_decode_ms: float
    p90_decode_ms: float
    p99_decode_ms: float
    # Per-request decode throughput (output tok/s) — P10 = slow tail
    mean_decode_throughput: float
    median_decode_throughput: float
    p10_decode_throughput: float
    p25_decode_throughput: float


def calculate_metrics(
    input_requests: list[DatasetRow],
    outputs: list[RequestOutput],
    duration_s: float,
) -> BenchmarkMetrics:
    ttfts: list[float] = []
    tpots: list[float] = []
    itls: list[float] = []
    e2e_latencies: list[float] = []
    prefill_throughputs: list[float] = []
    decode_times: list[float] = []
    decode_throughputs: list[float] = []
    total_prefill_tokens = 0
    total_prefill_time = 0.0
    total_decode_tokens = 0
    total_decode_time = 0.0
    total_input = 0
    total_output = 0
    completed = 0

    for i, out in enumerate(outputs):
        if not out.success:
            continue
        completed += 1
        total_input += input_requests[i].prompt_len
        total_output += out.output_len
        ttfts.append(out.ttft)
        if out.output_len > 1:
            tpots.append((out.latency - out.ttft) / (out.output_len - 1))
        itls.extend(out.itl)
        e2e_latencies.append(out.latency)
        # Prefill throughput: input tokens processed during TTFT
        if out.ttft > 0:
            prefill_throughputs.append(input_requests[i].prompt_len / out.ttft)
            total_prefill_tokens += input_requests[i].prompt_len
            total_prefill_time += out.ttft
        # Decode phase: time and throughput after first token
        decode_time = out.latency - out.ttft
        if decode_time > 0 and out.output_len > 1:
            decode_times.append(decode_time)
            decode_throughputs.append((out.output_len - 1) / decode_time)
            total_decode_tokens += out.output_len - 1
            total_decode_time += decode_time

    def _safe(func, data, *a, default=0.0):
        return float(func(data, *a)) if data else default

    return BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=completed / duration_s,
        input_throughput=total_input / duration_s,
        output_throughput=total_output / duration_s,
        total_throughput=(total_input + total_output) / duration_s,
        mean_ttft_ms=_safe(np.mean, ttfts) * 1000,
        median_ttft_ms=_safe(np.median, ttfts) * 1000,
        p50_ttft_ms=_safe(np.percentile, ttfts, 50) * 1000,
        std_ttft_ms=_safe(np.std, ttfts) * 1000,
        p90_ttft_ms=_safe(np.percentile, ttfts, 90) * 1000,
        p99_ttft_ms=_safe(np.percentile, ttfts, 99) * 1000,
        mean_tpot_ms=_safe(np.mean, tpots) * 1000,
        median_tpot_ms=_safe(np.median, tpots) * 1000,
        std_tpot_ms=_safe(np.std, tpots) * 1000,
        p90_tpot_ms=_safe(np.percentile, tpots, 90) * 1000,
        p99_tpot_ms=_safe(np.percentile, tpots, 99) * 1000,
        mean_itl_ms=_safe(np.mean, itls) * 1000,
        median_itl_ms=_safe(np.median, itls) * 1000,
        p50_itl_ms=_safe(np.percentile, itls, 50) * 1000,
        std_itl_ms=_safe(np.std, itls) * 1000,
        p90_itl_ms=_safe(np.percentile, itls, 90) * 1000,
        p95_itl_ms=_safe(np.percentile, itls, 95) * 1000,
        p99_itl_ms=_safe(np.percentile, itls, 99) * 1000,
        max_itl_ms=_safe(np.max, itls) * 1000,
        mean_e2e_latency_ms=_safe(np.mean, e2e_latencies) * 1000,
        median_e2e_latency_ms=_safe(np.median, e2e_latencies) * 1000,
        std_e2e_latency_ms=_safe(np.std, e2e_latencies) * 1000,
        p90_e2e_latency_ms=_safe(np.percentile, e2e_latencies, 90) * 1000,
        p99_e2e_latency_ms=_safe(np.percentile, e2e_latencies, 99) * 1000,
        concurrency=_safe(np.sum, e2e_latencies) / duration_s if duration_s > 0 else 0.0,
        agg_prefill_throughput=total_prefill_tokens / total_prefill_time if total_prefill_time > 0 else 0.0,
        agg_decode_throughput=total_decode_tokens / total_decode_time if total_decode_time > 0 else 0.0,
        mean_prefill_throughput=_safe(np.mean, prefill_throughputs),
        median_prefill_throughput=_safe(np.median, prefill_throughputs),
        p10_prefill_throughput=_safe(np.percentile, prefill_throughputs, 10),
        p25_prefill_throughput=_safe(np.percentile, prefill_throughputs, 25),
        mean_decode_ms=_safe(np.mean, decode_times) * 1000,
        median_decode_ms=_safe(np.median, decode_times) * 1000,
        p90_decode_ms=_safe(np.percentile, decode_times, 90) * 1000,
        p99_decode_ms=_safe(np.percentile, decode_times, 99) * 1000,
        mean_decode_throughput=_safe(np.mean, decode_throughputs),
        median_decode_throughput=_safe(np.median, decode_throughputs),
        p10_decode_throughput=_safe(np.percentile, decode_throughputs, 10),
        p25_decode_throughput=_safe(np.percentile, decode_throughputs, 25),
    )


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

async def run_benchmark(
    backend: str,
    api_url: str,
    model: str,
    input_requests: list[DatasetRow],
    request_rate: float,
    max_concurrency: int | None,
    num_warmup: int = 3,
) -> tuple[list[RequestOutput], float]:
    request_func = BACKEND_FUNCS[backend]

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request(inp: RequestInput, pbar: tqdm | None):
        if semaphore is None:
            return await request_func(inp, pbar)
        async with semaphore:
            return await request_func(inp, pbar)

    # Warmup — send enough requests to trigger CUDA graph compilation
    # for common batch sizes, reducing variance in early benchmark requests.
    print(f"Sending {num_warmup} warmup requests...")
    for w in range(num_warmup):
        warmup_row = input_requests[w % len(input_requests)]
        warmup_input = RequestInput(
            model=model,
            prompt=warmup_row.prompt,
            api_url=api_url,
            prompt_len=warmup_row.prompt_len,
            output_len=min(warmup_row.output_len, 128),
        )
        warmup_out = await request_func(warmup_input, pbar=None)
        if not warmup_out.success:
            raise RuntimeError(f"Warmup {w+1} failed: {warmup_out.error}")
    print("Warmup done.\n")

    # Main benchmark
    tasks: list[asyncio.Task] = []
    pbar = tqdm(total=len(input_requests))
    benchmark_start = time.perf_counter()

    async for row in _get_requests(input_requests, request_rate):
        inp = RequestInput(
            model=model,
            prompt=row.prompt,
            api_url=api_url,
            prompt_len=row.prompt_len,
            output_len=row.output_len,
        )
        tasks.append(asyncio.create_task(limited_request(inp, pbar)))

    outputs: list[RequestOutput] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - benchmark_start
    pbar.close()

    return outputs, duration


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_results(
    metrics: BenchmarkMetrics,
    duration: float,
    backend: str,
    request_rate: float,
    max_concurrency: int | None,
) -> dict[str, Any]:
    W = 42  # label width

    print(f"\n{'=' * 55}")
    print(f"{'Serving Benchmark Result':^55}")
    print(f"{'=' * 55}")
    print(f"{'Backend:':<{W}} {backend}")
    print(f"{'Request rate:':<{W}} {request_rate}")
    print(f"{'Max concurrency:':<{W}} {max_concurrency or 'unlimited'}")
    print(f"{'Successful requests:':<{W}} {metrics.completed}")
    print(f"{'Benchmark duration (s):':<{W}} {duration:.2f}")
    print(f"{'Total input tokens:':<{W}} {metrics.total_input}")
    print(f"{'Total output tokens:':<{W}} {metrics.total_output}")
    print(f"{'Request throughput (req/s):':<{W}} {metrics.request_throughput:.2f}")
    print(f"{'Input token throughput (tok/s):':<{W}} {metrics.input_throughput:.2f}")
    print(f"{'Output token throughput (tok/s):':<{W}} {metrics.output_throughput:.2f}")
    print(f"{'Total token throughput (tok/s):':<{W}} {metrics.total_throughput:.2f}")
    print(f"{'Concurrency:':<{W}} {metrics.concurrency:.2f}")

    print(f"{'-' * 55}")
    print(f"{'End-to-End Latency':^55}")
    print(f"{'-' * 55}")
    print(f"{'Mean E2E Latency (ms):':<{W}} {metrics.mean_e2e_latency_ms:.2f}")
    print(f"{'Median E2E Latency (ms):':<{W}} {metrics.median_e2e_latency_ms:.2f}")
    print(f"{'P90 E2E Latency (ms):':<{W}} {metrics.p90_e2e_latency_ms:.2f}")
    print(f"{'P99 E2E Latency (ms):':<{W}} {metrics.p99_e2e_latency_ms:.2f}")

    print(f"{'-' * 55}")
    print(f"{'Time to First Token':^55}")
    print(f"{'-' * 55}")
    print(f"{'Mean TTFT (ms):':<{W}} {metrics.mean_ttft_ms:.2f}")
    print(f"{'Median TTFT (ms):':<{W}} {metrics.median_ttft_ms:.2f}")
    print(f"{'P50 TTFT (ms):':<{W}} {metrics.p50_ttft_ms:.4f}")
    print(f"{'P90 TTFT (ms):':<{W}} {metrics.p90_ttft_ms:.2f}")
    print(f"{'P99 TTFT (ms):':<{W}} {metrics.p99_ttft_ms:.2f}")

    print(f"{'-' * 55}")
    print(f"{'Time per Output Token (excl. 1st token)':^55}")
    print(f"{'-' * 55}")
    print(f"{'Mean TPOT (ms):':<{W}} {metrics.mean_tpot_ms:.2f}")
    print(f"{'Median TPOT (ms):':<{W}} {metrics.median_tpot_ms:.2f}")
    print(f"{'P90 TPOT (ms):':<{W}} {metrics.p90_tpot_ms:.2f}")
    print(f"{'P99 TPOT (ms):':<{W}} {metrics.p99_tpot_ms:.2f}")

    print(f"{'-' * 55}")
    print(f"{'Prefill Phase':^55}")
    print(f"{'-' * 55}")
    print(f"{'Aggregate Prefill Throughput (tok/s):':<{W}} {metrics.agg_prefill_throughput:.2f}")
    print(f"{'Mean per-req Prefill Tput (tok/s):':<{W}} {metrics.mean_prefill_throughput:.2f}")
    print(f"{'Median per-req Prefill Tput (tok/s):':<{W}} {metrics.median_prefill_throughput:.2f}")
    print(f"{'P10 per-req Prefill Tput (tok/s):':<{W}} {metrics.p10_prefill_throughput:.2f}")
    print(f"{'P25 per-req Prefill Tput (tok/s):':<{W}} {metrics.p25_prefill_throughput:.2f}")

    print(f"{'-' * 55}")
    print(f"{'Decode Phase':^55}")
    print(f"{'-' * 55}")
    print(f"{'Aggregate Decode Throughput (tok/s):':<{W}} {metrics.agg_decode_throughput:.2f}")
    print(f"{'Mean per-req Decode Tput (tok/s):':<{W}} {metrics.mean_decode_throughput:.2f}")
    print(f"{'Median per-req Decode Tput (tok/s):':<{W}} {metrics.median_decode_throughput:.2f}")
    print(f"{'P10 per-req Decode Tput (tok/s):':<{W}} {metrics.p10_decode_throughput:.2f}")
    print(f"{'P25 per-req Decode Tput (tok/s):':<{W}} {metrics.p25_decode_throughput:.2f}")
    print(f"{'Mean Decode Time (ms):':<{W}} {metrics.mean_decode_ms:.2f}")
    print(f"{'Median Decode Time (ms):':<{W}} {metrics.median_decode_ms:.2f}")
    print(f"{'P90 Decode Time (ms):':<{W}} {metrics.p90_decode_ms:.2f}")
    print(f"{'P99 Decode Time (ms):':<{W}} {metrics.p99_decode_ms:.2f}")

    print(f"{'-' * 55}")
    print(f"{'Time Between Tokens (TBT)':^55}")
    print(f"{'-' * 55}")
    print(f"{'Mean TBT (ms):':<{W}} {metrics.mean_itl_ms:.2f}")
    print(f"{'Median TBT (ms):':<{W}} {metrics.median_itl_ms:.2f}")
    print(f"{'P50 TBT (ms):':<{W}} {metrics.p50_itl_ms:.4f}")
    print(f"{'P90 TBT (ms):':<{W}} {metrics.p90_itl_ms:.2f}")
    print(f"{'P95 TBT (ms):':<{W}} {metrics.p95_itl_ms:.2f}")
    print(f"{'P99 TBT (ms):':<{W}} {metrics.p99_itl_ms:.2f}")
    print(f"{'Max TBT (ms):':<{W}} {metrics.max_itl_ms:.2f}")
    print(f"{'=' * 55}")

    result = {
        "backend": backend,
        "request_rate": request_rate,
        "max_concurrency": max_concurrency,
        "duration": duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "total_throughput": metrics.total_throughput,
        "concurrency": metrics.concurrency,
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
        "p90_e2e_latency_ms": metrics.p90_e2e_latency_ms,
        "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "p50_ttft_ms": metrics.p50_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p90_ttft_ms": metrics.p90_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p90_tpot_ms": metrics.p90_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "p50_tbt_ms": metrics.p50_itl_ms,
        "median_tbt_ms": metrics.median_itl_ms,
        "p90_itl_ms": metrics.p90_itl_ms,
        "p95_itl_ms": metrics.p95_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "max_itl_ms": metrics.max_itl_ms,
        "mean_prefill_throughput": metrics.mean_prefill_throughput,
        "median_prefill_throughput": metrics.median_prefill_throughput,
        "p10_prefill_throughput": metrics.p10_prefill_throughput,
        "p25_prefill_throughput": metrics.p25_prefill_throughput,
        "agg_prefill_throughput": metrics.agg_prefill_throughput,
        "mean_decode_ms": metrics.mean_decode_ms,
        "median_decode_ms": metrics.median_decode_ms,
        "p90_decode_ms": metrics.p90_decode_ms,
        "p99_decode_ms": metrics.p99_decode_ms,
        "mean_decode_throughput": metrics.mean_decode_throughput,
        "median_decode_throughput": metrics.median_decode_throughput,
        "p10_decode_throughput": metrics.p10_decode_throughput,
        "p25_decode_throughput": metrics.p25_decode_throughput,
        "agg_decode_throughput": metrics.agg_decode_throughput,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def set_ulimit(target: int = 65535) -> None:
    current_soft, current_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if current_soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, current_hard))
        except ValueError as exc:
            print(f"Warning: could not raise RLIMIT_NOFILE: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone SGLang serving benchmark (no yolo dependency).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Server
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(BACKEND_FUNCS.keys()),
        default="sglang",
        help="Backend type: 'sglang' for /generate, 'openai' for /v1/completions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model name sent in the request (needed for openai backend, e.g. Qwen/Qwen2.5-7B-Instruct)",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to JSONL dataset (each line: {prompt, prompt_len, output_len})",
    )
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to use")

    # Random dataset (when --dataset is not given)
    parser.add_argument("--random-input-len", type=int, default=512, help="Input length for random prompts")
    parser.add_argument("--random-output-len", type=int, default=256, help="Output length for random prompts")

    # Tokenizer
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HuggingFace tokenizer path or model ID",
    )

    # Benchmark control
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate (req/s). Use 'inf' for max throughput (default: inf)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent in-flight requests",
    )

    # Stability
    parser.add_argument("--num-warmup", type=int, default=3, help="Number of warmup requests before benchmark")
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        default=False,
        help="Call /flush_cache on the server before benchmarking to clear KV cache",
    )

    # Output
    parser.add_argument("--output-file", type=str, default=None, help="Save results JSON to this file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Load dataset
    if args.dataset:
        print(f"Loading dataset from {args.dataset}...")
        input_requests = load_jsonl_dataset(args.dataset, args.num_prompts)
    else:
        print(f"Generating {args.num_prompts} random prompts (input={args.random_input_len}, output={args.random_output_len})...")
        input_requests = generate_random_dataset(
            tokenizer, args.num_prompts, args.random_input_len, args.random_output_len,
        )
    print(f"Loaded {len(input_requests)} prompts.")

    # Build API URL
    if args.backend == "sglang":
        api_url = f"http://{args.host}:{args.port}/generate"
    elif args.backend == "openai":
        api_url = f"http://{args.host}:{args.port}/v1/completions"
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    print(f"Targeting {api_url}")

    # Check server health
    health_url = f"http://{args.host}:{args.port}/v1/models"
    print(f"Checking server health at {health_url}...")
    try:
        resp = requests.get(health_url, timeout=10)
        if resp.status_code == 200:
            print("Server is ready.")
        else:
            print(f"Warning: health check returned HTTP {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to server at {args.host}:{args.port}")
        return 1

    # Flush KV cache to start from a clean state
    if args.flush_cache:
        flush_url = f"http://{args.host}:{args.port}/flush_cache"
        print(f"Flushing server cache at {flush_url}...")
        try:
            resp = requests.post(flush_url, timeout=30)
            if resp.status_code == 200:
                print("Cache flushed.")
            else:
                print(f"Warning: flush_cache returned HTTP {resp.status_code}")
        except Exception as e:
            print(f"Warning: flush_cache failed: {e}")

    # Run benchmark
    print(f"\nStarting benchmark: backend={args.backend}, rate={args.request_rate}, max_concurrency={args.max_concurrency}")
    outputs, duration = asyncio.run(
        run_benchmark(
            backend=args.backend,
            api_url=api_url,
            model=args.model,
            input_requests=input_requests,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            num_warmup=args.num_warmup,
        )
    )

    # Calculate and print metrics
    metrics = calculate_metrics(input_requests, outputs, duration)

    succeeded = sum(1 for o in outputs if o.success)
    failed = sum(1 for o in outputs if not o.success)
    if failed > 0:
        print(f"\nWarning: {failed}/{len(outputs)} requests failed.")
        for out in outputs:
            if not out.success and out.error:
                print(f"  Error: {out.error[:200]}")
                break

    result = print_results(metrics, duration, args.backend, args.request_rate, args.max_concurrency)

    # Save results
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    return 0 if succeeded > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
