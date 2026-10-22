"""Summarize GPUNetIO submission timing, counts, and dispatch timelines from buffered EP traces."""

import argparse
from collections import Counter, defaultdict
import csv
from glob import glob
from math import isfinite
from pathlib import Path
import re
import sys

FIELDS = "rank operation block invocation epoch thread kind peer queue begin_ns end_ns".split() + [
    f"v{index}" for index in range(8)
]
IDENTITY = "rank operation block invocation epoch thread peer queue end_ns".split()
METRICS = (
    "submit_ns",
    "lock_wait_ns",
    "lock_held_ns",
    "index_or_fence_ns",
    "first_db_or_proxy_ns",
    "dbrec_ns",
    "second_db_ns",
    "other_ns",
)


def load_samples(path, counts_only=False, timeline=False):
    path = Path(path)
    blocks, parents, details = {}, {}, {}
    counts, ranks = Counter(), set()
    count_records = []
    timeline_records = []
    with path.open(newline="") as stream:

        def lines():
            for line in stream:
                if line.startswith("# operation="):
                    match = re.fullmatch(r"# operation=(\d+) block=(\d+) attempted=(\d+) dropped=(\d+)", line.strip())
                    if match is None:
                        raise ValueError(f"{path}: malformed block metadata")
                    operation, block, attempted, dropped = map(int, match.groups())
                    if dropped or (operation, block) in blocks:
                        raise ValueError(f"{path}: dropped={dropped} or repeated block metadata")
                    blocks[operation, block] = attempted
                elif line.strip() and not line.startswith("#"):
                    yield line

        reader = csv.DictReader(lines())
        if reader.fieldnames != FIELDS:
            raise ValueError(f"{path}: empty file or unsupported CSV header")
        for row in reader:
            try:
                if None in row:
                    raise ValueError("extra columns")
                record = {field: int(row[field]) for field in FIELDS}
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}: malformed CSV record") from error
            counts[record["operation"], record["block"]] += 1
            ranks.add(record["rank"])
            if timeline:
                if record["operation"] != 0:
                    continue
                if record["kind"] == 13:
                    raise ValueError(
                        f"{path}: counts-only records have no timeline; disable MSCCLPP_EP_TIMING_SUBMIT_COUNTS"
                    )
                if (
                    any(record[field] < 0 for field in FIELDS if field not in ("peer", "queue"))
                    or record["invocation"] < 1
                    or record["end_ns"] < record["begin_ns"]
                ):
                    raise ValueError(f"{path}: invalid timeline record")
                record["file"] = str(path.resolve())
                timeline_records.append(record)
                continue
            if counts_only and (record["operation"] != 0 or record["kind"] != 13):
                raise ValueError(f"{path}: expected counts-only trace; enable MSCCLPP_EP_TIMING_SUBMIT_COUNTS=1")
            if record["operation"] != 0 or record["kind"] not in ((13,) if counts_only else (9, 10, 11, 12)):
                continue
            if (
                any(record[field] < 0 for field in FIELDS)
                or record["invocation"] < 1
                or record["end_ns"] < record["begin_ns"]
            ):
                raise ValueError(f"{path}: invalid posting identity, timestamp or value")
            if counts_only:
                handler, issued, acquired, bypass, requested, advanced, extra, coalesced = (
                    record[f"v{index}"] for index in range(8)
                )
                if any(value not in (0, 1) for value in (issued, acquired, bypass, coalesced)) or requested != 3:
                    raise ValueError(f"{path}: invalid submit count flags or requested WQEs")
                if record["begin_ns"] != 0 or record["end_ns"] != 0 or extra > advanced:
                    raise ValueError(f"{path}: inconsistent count record")
                if handler == 2:
                    if acquired + bypass != 1 or issued > acquired or (bypass and not coalesced):
                        raise ValueError(f"{path}: inconsistent lock/bypass counts")
                elif acquired or bypass or coalesced:
                    raise ValueError(f"{path}: non-DB handler has lock/coalescing counts")
                if handler in (2, 10):
                    if bool(advanced) != bool(issued):
                        raise ValueError(f"{path}: publication and WQE advance disagree")
                elif not issued or advanced:
                    raise ValueError(f"{path}: invalid proxy request count")
                if extra and not coalesced:
                    raise ValueError(f"{path}: extra-ready WQEs without coalescing")
                record["file"] = str(path.resolve())
                count_records.append(record)
                continue
            key = tuple(record[field] for field in IDENTITY)
            target = details if record["kind"] == 12 else parents
            if key in target:
                raise ValueError(f"{path}: ambiguous duplicate posting identity")
            target[key] = record
    if len(ranks) != 1 or dict(counts) != blocks:
        raise ValueError(f"{path}: truncated records, mixed ranks or missing block metadata")
    if timeline:
        if not timeline_records:
            raise ValueError(f"{path}: no dispatch timeline records")
        return timeline_records
    if counts_only:
        if not count_records:
            raise ValueError(f"{path}: no kind=13 submit counts")
        return count_records
    if not details:
        raise ValueError(f"{path}: no kind=12 records; enable MSCCLPP_EP_TIMING_SUBMIT_EVERY")
    samples = []
    for key, detail in details.items():
        parent = parents.get(key)
        span = detail["end_ns"] - detail["begin_ns"]
        if parent is None:
            raise ValueError(f"{path}: submit detail has no matching post/chunk")
        if (
            detail["begin_ns"] < parent["begin_ns"]
            or span != parent["v3"]
            or sum(parent[f"v{index}"] for index in range(4)) != parent["end_ns"] - parent["begin_ns"]
        ):
            raise ValueError(f"{path}: inconsistent submit/post span")
        if detail["v7"] not in (0, 1):
            raise ValueError(f"{path}: invalid publication flag")
        if parent["kind"] == 9:
            mode, rows, size = "token", 1, parent["v5"]
        elif parent["kind"] == 10:
            mode, rows, size = "batch", parent["v5"], parent["v6"]
        else:
            modes = {0: "full", 1: "control_tail", 2: "producer_tail"}
            if parent["v7"] not in modes:
                raise ValueError(f"{path}: invalid chunk classification")
            mode, rows, size = modes[parent["v7"]], parent["v5"], parent["v6"]
        if rows <= 0 or size <= 0:
            raise ValueError(f"{path}: invalid posting extent")
        regular = detail["v0"] == 2
        covered = detail["v1"] + detail["v6"] if regular else detail["v2"] + detail["v3"]
        if covered > span or (regular and sum(detail[f"v{index}"] for index in (2, 3, 4, 5)) > detail["v6"]):
            raise ValueError(f"{path}: inconsistent submission stages")
        sample = {field: detail[field] for field in ("rank", "peer", "queue", "invocation", "epoch")}
        sample.update(
            file=str(path.resolve()),
            mode=mode,
            rows=rows,
            bytes=size,
            handler=detail["v0"],
            issued=detail["v7"],
            submit_ns=span,
            lock_wait_ns=detail["v1"] if regular else None,
            lock_held_ns=detail["v6"] if regular else None,
            index_or_fence_ns=detail["v2"],
            first_db_or_proxy_ns=detail["v3"],
            dbrec_ns=detail["v4"] if regular else None,
            second_db_ns=detail["v5"] if regular else None,
            other_ns=span - covered,
        )
        samples.append(sample)
    return samples


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def report_counts(records):
    groups = defaultdict(list)
    keys = ("file", "rank", "peer", "queue", "handler", "coalesced")
    for record in records:
        groups[(record["file"], record["rank"], record["peer"], record["queue"], record["v0"], record["v7"])].append(
            record
        )
    print(
        "# Exact counts for retained dispatch putBatched3 calls only; compare runs with matching workload/invocation coverage."
    )
    print(
        "# Not all QP traffic: markers, count writes and combine calls are omitted; observed publications may cover their WQEs."
    )
    print(
        "# Publication sequences: handler 2 rings twice, handler 10 once; other handlers issue proxy requests, not observed NIC doorbells."
    )
    print(
        "# Extra-new WQEs excludes previously published prefixes. Counts collection still adds overhead; not a performance benchmark."
    )
    print(
        "# Warmup/capture calls are included unless filtered. Check all rank EPTIMING saved/dropped=0 logs; CSV has no footer."
    )
    fields = [
        *keys,
        "invocation_ordinals",
        "first_invocation",
        "last_invocation",
        "calls",
        "requested_payload_wqes",
        "publication_sequences",
        "doorbell_calls",
        "proxy_requests",
        "lock_acquisitions",
        "watermark_bypasses",
        "covered_after_lock",
        "published_wqes",
        "extra_new_wqes",
        "wqes_per_publication",
        "bypass_pct",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for key, samples in sorted(groups.items()):
        result = dict(zip(keys, key))
        totals = [sum(record[f"v{index}"] for record in samples) for index in range(8)]
        handler = key[4]
        doorbell = handler in (2, 10)
        invocations = {record["invocation"] for record in samples}
        result.update(
            invocation_ordinals=len(invocations),
            first_invocation=min(invocations),
            last_invocation=max(invocations),
            calls=len(samples),
            requested_payload_wqes=totals[4],
            publication_sequences=totals[1] if doorbell else "-",
            doorbell_calls=totals[1] * (2 if handler == 2 else 1) if doorbell else "-",
            proxy_requests=0 if doorbell else totals[1],
            lock_acquisitions=totals[2],
            watermark_bypasses=totals[3],
            covered_after_lock=totals[2] - totals[1] if handler == 2 else "-",
            published_wqes=totals[5] if doorbell else "-",
            extra_new_wqes=totals[6],
            wqes_per_publication=f"{totals[5]/totals[1]:.3f}" if doorbell and totals[1] else "-",
            bypass_pct=f"{100*totals[3]/len(samples):.3f}",
        )
        writer.writerow(result)


def load_count_run(patterns, expected_mode, skip_invocations=0, expected_ranks=None):
    records = []
    files = {}
    seen = set()
    for pattern in patterns:
        matches = [pattern] if Path(pattern).is_file() else sorted(glob(pattern))
        if not matches:
            raise ValueError(f"no files match {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if path in seen:
                raise ValueError(f"duplicate input: {path}")
            seen.add(path)
            loaded = load_samples(path, counts_only=True)
            rank = loaded[0]["rank"]
            if rank in files:
                raise ValueError(f"multiple files for rank {rank} in one run; select exactly one context/run per rank")
            files[rank] = str(path)
            retained = 0
            for record in loaded:
                if record["v0"] == 2 and record["v7"] != expected_mode:
                    raise ValueError(f"{path}: expected coalescing={expected_mode} for GPU_SM_DB records")
                if record["invocation"] > skip_invocations:
                    records.append(record)
                    retained += 1
            if retained == 0:
                raise ValueError(f"{path}: no count records remain after filtering")
    if expected_ranks is not None and set(files) != set(range(expected_ranks)):
        raise ValueError(f"expected ranks 0..{expected_ranks-1}, found {sorted(files)}")
    if not records:
        raise ValueError("no count records remain after filtering")
    return records, files


def compare_counts(baseline, candidate, by_rank=False):
    fields = ("rank", "v0") if by_rank else ("rank", "peer", "queue", "v0")
    grouped = []
    for records in (baseline, candidate):
        groups = defaultdict(list)
        for record in records:
            groups[tuple(record[field] for field in fields)].append(record)
        grouped.append(groups)

    def coverage(samples):
        return Counter((row["peer"], row["queue"], row["invocation"], row["epoch"], row["v4"]) for row in samples)

    def totals(samples):
        if not samples:
            return {}
        values = [sum(row[f"v{index}"] for row in samples) for index in range(8)]
        handler = samples[0]["v0"]
        doorbell = handler in (2, 10)
        return {
            "calls": len(samples),
            "requested_wqes": values[4],
            "publications": values[1] if doorbell else "-",
            "doorbells": values[1] * (2 if handler == 2 else 1) if doorbell else "-",
            "proxy_requests": 0 if doorbell else values[1],
            "locks": values[2],
            "bypasses": values[3],
            "covered_after_lock": values[2] - values[1] if handler == 2 else "-",
            "published_wqes": values[5] if doorbell else "-",
            "extra_new_wqes": values[6],
            "wqes_per_publication": values[5] / values[1] if doorbell and values[1] else "-",
        }

    report = []
    for key in sorted(grouped[0].keys() | grouped[1].keys()):
        before, after = grouped[0].get(key, []), grouped[1].get(key, [])
        status = (
            "missing_baseline"
            if not before
            else (
                "missing_candidate"
                if not after
                else ("matched" if coverage(before) == coverage(after) else "coverage_mismatch")
            )
        )
        row = {
            "rank": key[0],
            "peer": "-" if by_rank else key[1],
            "queue": "-" if by_rank else key[2],
            "handler": key[-1],
            "status": status,
            "baseline_file": before[0]["file"] if before else "-",
            "candidate_file": after[0]["file"] if after else "-",
        }
        old, new = totals(before), totals(after)
        for metric in (
            "calls",
            "requested_wqes",
            "publications",
            "doorbells",
            "proxy_requests",
            "locks",
            "bypasses",
            "covered_after_lock",
            "published_wqes",
            "extra_new_wqes",
            "wqes_per_publication",
        ):
            row[f"{metric}_off"] = old.get(metric, "-")
            row[f"{metric}_on"] = new.get(metric, "-")
        for metric in ("publications", "locks"):
            old_value, new_value = old.get(metric), new.get(metric)
            matched = status == "matched" and isinstance(old_value, int) and isinstance(new_value, int)
            row[f"{metric}_saved"] = old_value - new_value if matched else "-"
            row[f"{metric}_reduction_pct"] = 100 * (old_value - new_value) / old_value if matched and old_value else "-"
        report.append(row)
    return report


def report_comparison(report, by_rank=False):
    print(
        "# Coalescing OFF vs ON; positive reductions mean fewer publications/locks. Scope: dispatch putBatched3 only."
    )
    print("# Rank summaries sum counters per handler; ratios use summed totals, not averages of per-QP ratios.")
    print(
        "# Coverage matches recorded peer/QP/invocation/epoch call counts, not payload bytes or unrecorded warmup/workload settings."
    )
    print("# Missing or mismatched groups retain raw counts but have no deltas; '-' is unavailable, not zero.")
    print(
        "# Published frontier WQEs may include other QP traffic. These counts do not identify the kernel critical path."
    )
    print("# Validate each rank's EPTIMING saved/dropped=0 log; the CSV format has no completion footer.")
    writer = csv.DictWriter(sys.stdout, fieldnames=list(report[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in report:
        writer.writerow({field: f"{value:.3f}" if isinstance(value, float) else value for field, value in row.items()})
    matched = [row for row in report if row["status"] == "matched" and row["handler"] == 2]
    for metric in ("publications", "locks"):
        decreased = sum(row[f"{metric}_saved"] > 0 for row in matched)
        unchanged = sum(row[f"{metric}_saved"] == 0 for row in matched)
        increased = sum(row[f"{metric}_saved"] < 0 for row in matched)
        print(
            f"# GPU_SM_DB {'rank' if by_rank else 'QP'} groups: {metric} decreased={decreased} unchanged={unchanged} increased={increased}"
        )
    mismatches = sum(row["status"] != "matched" for row in report)
    if mismatches:
        print(
            f"WARNING: {mismatches} groups have missing or mismatched call coverage; reduction percentages suppressed",
            file=sys.stderr,
        )


TIMELINE_STAGES = ("producer", "notify", "payload_post", "marker", "drain", "arrival_wait", "receive_handler")
CORE_STAGES = ("producer", "payload_post", "drain", "arrival_wait")


def dispatch_timeline(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["file"], record["rank"], record["invocation"], record["epoch"])].append(record)
    report = []
    for identity, samples in sorted(grouped.items()):
        label = f"{identity[0]} rank={identity[1]} invocation={identity[2]} epoch={identity[3]}"
        kinds = defaultdict(list)
        for sample in samples:
            kinds[sample["kind"]].append(sample)
        if len(kinds[15]) != 1:
            raise ValueError(
                f"{label}: missing/duplicate DispatchTopology; collect a fresh trace with the timeline patch"
            )
        topology = kinds[15][0]
        ranks, first, producers, groups, control, queues, remote_mask, asynchronous = (
            topology[f"v{index}"] for index in range(8)
        )
        if not (
            1 <= ranks <= 64
            and 0 <= identity[1] < ranks
            and asynchronous in (0, 1)
            and first == (ranks if asynchronous else 1)
            and producers > 0
            and groups in (1, 2)
            and control == first + producers
            and ranks <= control
            and 0 <= queues <= 64
            and remote_mask < (1 << ranks)
            and not (remote_mask & (1 << identity[1]))
            and (queues > 0 or remote_mask == 0)
            and topology["block"] == 0
            and topology["thread"] == 0
        ):
            raise ValueError(f"{label}: invalid CTA topology")
        remote = {peer for peer in range(ranks) if remote_mask & (1 << peer)}

        def require_keys(rows, fields, expected, name):
            actual = [tuple(row[field] for field in fields) for row in rows]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError(f"{label}: incomplete/duplicate {name} coverage")

        require_keys(kinds[0], ("block", "thread"), {(block, 0) for block in range(control + 1)}, "CTA completion")
        if any(row["v6"] != control + 1 for row in kinds[0]):
            raise ValueError(f"{label}: inconsistent grid size")
        require_keys(
            kinds[1],
            ("block", "thread"),
            {(block, group * (16 // groups) * 32) for block in range(first, control) for group in range(groups)},
            "producer warp-group",
        )
        require_keys(
            kinds[2],
            ("block", "thread", "peer"),
            {(control, peer, peer) for peer in range(ranks)},
            "count notification",
        )
        require_keys(
            kinds[14], ("block", "thread", "peer"), {(peer, 0, peer) for peer in range(ranks)}, "receive handler"
        )
        for row in kinds[14]:
            path = 0 if row["peer"] == identity[1] else 2 if row["peer"] in remote else 1
            if row["v0"] != path:
                raise ValueError(f"{label}: receive path disagrees with topology")
        expected_queues = {(peer, queue) for peer in remote for queue in range(queues)}
        queue_records = {}
        for action, name in ((0, "marker"), (1, "drain"), (2, "arrival_wait")):
            rows = [row for row in kinds[3] if row["v0"] == action]
            require_keys(rows, ("peer", "queue"), expected_queues, name)
            if action == 2 and any(row["block"] != row["peer"] for row in rows):
                raise ValueError(f"{label}: arrival wait is not on its source CTA")
            queue_records[name] = rows
        if any(row["v0"] not in (0, 1, 2) for row in kinds[3]):
            raise ValueError(f"{label}: unknown dispatch queue action")
        payload = kinds[9] + kinds[10] + kinds[11]
        for row in payload:
            if (row["peer"], row["queue"]) not in expected_queues or sum(row[f"v{index}"] for index in range(4)) != row[
                "end_ns"
            ] - row["begin_ns"]:
                raise ValueError(f"{label}: invalid payload posting record")
        stages = {
            "producer": kinds[1],
            "notify": kinds[2],
            "payload_post": payload,
            **queue_records,
            "receive_handler": kinds[14],
        }
        origin = min(
            row["begin_ns"] for row in kinds[0] + kinds[15] + [event for rows in stages.values() for event in rows]
        )
        latest = {stage: max(rows, key=lambda row: row["end_ns"]) for stage, rows in stages.items() if rows}
        core_end = max(latest[stage]["end_ns"] for stage in CORE_STAGES if stage in latest)
        work_end = max(row["end_ns"] for row in latest.values())
        observed_end = max(row["end_ns"] for row in samples)
        result = dict(zip(("file", "rank", "invocation", "epoch"), identity))
        result.update(
            async_dispatch=asynchronous,
            blocks=control + 1,
            remote_peers=len(remote),
            qps_per_peer=queues,
            observed_span_us=(observed_end - origin) / 1000,
            last_core_stage="|".join(
                stage for stage in CORE_STAGES if stage in latest and latest[stage]["end_ns"] == core_end
            ),
            last_observed_work="|".join(
                stage for stage in TIMELINE_STAGES if stage in latest and latest[stage]["end_ns"] == work_end
            ),
            after_last_work_us=(observed_end - work_end) / 1000,
        )
        for stage in TIMELINE_STAGES:
            row = latest.get(stage)
            result[f"{stage}_end_us"] = (row["end_ns"] - origin) / 1000 if row else "-"
            result[f"{stage}_span_us"] = (row["end_ns"] - row["begin_ns"]) / 1000 if row else "-"
            result[f"{stage}_location"] = (
                f"block={row['block']},thread={row['thread']},peer={row['peer']},qp={row['queue']}" if row else "-"
            )
        report.append(result)
    if not report:
        raise ValueError("no dispatch invocations remain after filtering")
    return report


def report_timeline(report):
    print("# GPU-local times in us from earliest observed dispatch timestamp; one row per file/rank/invocation/epoch.")
    print(
        "# last_core_stage compares producer, payload_post, outgoing drain and incoming arrival-wait endpoints; ties are preserved."
    )
    print(
        "# last_observed_work also includes count notification, markers and receive-handler return. stage_span is for the selected latest event, not a sum."
    )
    print(
        "# Producer spans include inline posting; arrival endpoints record GPU polling completion, not NIC arrival timestamps."
    )
    print(
        "# CTA completion and receive-handler records are thread-0 observations. observed_span is not exact kernel duration; after_last_work includes tracing/epilogue gaps."
    )
    print(
        "# Last observed stage is not proof of a hardware bottleneck. Do not compare absolute timestamps across GPUs or add overlapping stage spans."
    )
    print(
        "# Complete CTA/QP coverage is required; also check EPTIMING saved/dropped=0 logs because the CSV has no completion footer."
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=list(report[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in report:
        writer.writerow({field: f"{value:.3f}" if isinstance(value, float) else value for field, value in row.items()})
    for rank in sorted({row["rank"] for row in report}):
        counts = Counter(row["last_core_stage"] for row in report if row["rank"] == rank)
        print(
            f"# rank={rank} last_core_stage_counts="
            + ",".join(f"{stage}:{count}" for stage, count in sorted(counts.items()))
        )


COMPARISON_STAGES = ("producer", "payload_post", "marker", "drain", "arrival_wait", "receive_handler")
TIMELINE_GAPS = {
    "arrival_minus_payload_us": ("arrival_wait", "payload_post"),
    "drain_minus_payload_us": ("drain", "payload_post"),
    "receive_minus_arrival_us": ("receive_handler", "arrival_wait"),
    "arrival_minus_drain_us": ("arrival_wait", "drain"),
}
TIMELINE_METRICS = (
    "observed_span_us",
    *[f"{stage}_end_us" for stage in COMPARISON_STAGES],
    "drain_span_us",
    "arrival_wait_span_us",
    "after_last_work_us",
    *TIMELINE_GAPS,
)
TIMELINE_LOCATION = re.compile(r"block=(\d+),thread=(\d+),peer=(-?\d+),qp=(-?\d+)")


def load_timeline_report(path, skip_invocations=0, expected_ranks=None):
    if skip_invocations < 0 or (expected_ranks is not None and expected_ranks < 1):
        raise ValueError("skip count must be nonnegative and expected rank count must be positive")
    integers = ("rank", "invocation", "epoch", "async_dispatch", "blocks", "remote_peers", "qps_per_peer")
    required = {"file", *integers, "observed_span_us", "last_core_stage", "last_observed_work", "after_last_work_us"}
    required.update(f"{stage}_{suffix}" for stage in TIMELINE_STAGES for suffix in ("end_us", "span_us", "location"))
    rows, identities, rank_files = [], set(), {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader((line for line in stream if line.strip() and not line.startswith("#")), delimiter="\t")
        if reader.fieldnames is None or len(reader.fieldnames) != len(required) or set(reader.fieldnames) != required:
            raise ValueError(f"{path}: expected a saved --timeline TSV report, not raw CSV or a counts report")
        for row in reader:
            if None in row or any(value is None or value == "" for value in row.values()):
                raise ValueError(f"{path}: malformed timeline report row")
            for field in integers:
                row[field] = int(row[field])
            if (
                any(row[field] < 0 for field in integers)
                or row["rank"] >= 64
                or row["invocation"] < 1
                or row["async_dispatch"] not in (0, 1)
                or row["blocks"] < 1
                or row["remote_peers"] >= 64
                or row["qps_per_peer"] > 64
                or (row["remote_peers"] and not row["qps_per_peer"])
            ):
                raise ValueError(f"{path}: invalid timeline identity or topology")
            for field in required:
                if field.endswith("_us") and row[field] != "-":
                    row[field] = float(row[field])
                    if not isfinite(row[field]) or row[field] < 0:
                        raise ValueError(f"{path}: invalid time in {field}")
            if row["observed_span_us"] == "-" or row["after_last_work_us"] == "-":
                raise ValueError(f"{path}: missing observation span")
            if row["after_last_work_us"] > row["observed_span_us"] + 0.001:
                raise ValueError(f"{path}: tail gap exceeds observation span")
            for stage in TIMELINE_STAGES:
                end, span, location = (row[f"{stage}_{suffix}"] for suffix in ("end_us", "span_us", "location"))
                if "-" in (end, span, location):
                    if (end, span, location) != ("-", "-", "-") or stage in ("producer", "notify", "receive_handler"):
                        raise ValueError(f"{path}: missing/inconsistent {stage} fields")
                else:
                    match = TIMELINE_LOCATION.fullmatch(location)
                    if match is None:
                        raise ValueError(f"{path}: malformed {stage} location")
                    block, thread, peer, queue = map(int, match.groups())
                    if (
                        block >= row["blocks"]
                        or thread >= 512
                        or not -1 <= peer < 64
                        or not -1 <= queue < row["qps_per_peer"]
                    ):
                        raise ValueError(f"{path}: invalid {stage} location")
                    if span > end + 0.001 or end > row["observed_span_us"] + 0.001:
                        raise ValueError(f"{path}: inconsistent {stage} times")
                if stage in ("marker", "drain", "arrival_wait") and (end != "-") != bool(row["remote_peers"]):
                    raise ValueError(f"{path}: {stage} disagrees with remote topology")
            for field, choices in (("last_core_stage", CORE_STAGES), ("last_observed_work", TIMELINE_STAGES)):
                stages = row[field].split("|")
                if len(stages) != len(set(stages)) or any(
                    stage not in choices or row[f"{stage}_end_us"] == "-" for stage in stages
                ):
                    raise ValueError(f"{path}: invalid {field}")
            identity = tuple(row[field] for field in ("rank", "invocation", "epoch"))
            if identity in identities:
                raise ValueError(f"{path}: duplicate rank/invocation/epoch")
            identities.add(identity)
            if rank_files.setdefault(row["rank"], row["file"]) != row["file"]:
                raise ValueError(f"{path}: multiple trace contexts for rank {row['rank']}")
            if row["invocation"] > skip_invocations:
                rows.append(row)
    if not rows or {row["rank"] for row in rows} != set(rank_files):
        raise ValueError(f"{path}: no retained rows for one or more ranks")
    if expected_ranks is not None and set(rank_files) != set(range(expected_ranks)):
        raise ValueError(f"{path}: expected ranks 0..{expected_ranks-1}, found {sorted(rank_files)}")
    return rows


def timeline_comparison_metrics(row):
    metrics = {field: row[field] for field in TIMELINE_METRICS if field not in TIMELINE_GAPS}
    for field, (later, earlier) in TIMELINE_GAPS.items():
        end, begin = row[f"{later}_end_us"], row[f"{earlier}_end_us"]
        metrics[field] = end - begin if end != "-" and begin != "-" else "-"
    return metrics


def compare_timeline_rows(baseline, candidate, top=5, by_rank=False):
    if top < 1:
        raise ValueError("--top must be positive")
    if not baseline or not candidate:
        raise ValueError("both timeline reports must contain rows")
    if {row["rank"] for row in baseline} != {row["rank"] for row in candidate}:
        raise ValueError("timeline rank sets differ")
    topology_fields = ("async_dispatch", "blocks", "remote_peers", "qps_per_peer")
    identifiers = ("file", "rank", "invocation", "epoch", "last_core_stage", "last_observed_work") + topology_fields
    locations = tuple(f"{stage}_location" for stage in COMPARISON_STAGES)
    order = lambda row: (-row["observed_span_us"], row["rank"], row["invocation"], row["epoch"], row["file"])
    report = []
    scopes = sorted({row["rank"] for row in baseline}) if by_rank else [None]
    for rank in scopes:
        old = sorted((row for row in baseline if rank is None or row["rank"] == rank), key=order)
        new = sorted((row for row in candidate if rank is None or row["rank"] == rank), key=order)
        coverage = lambda rows: Counter(row["rank"] for row in rows)
        topology = lambda rows: {(row["rank"], *(row[field] for field in topology_fields)) for row in rows}
        issues = []
        if coverage(old) != coverage(new):
            issues.append("sample_count_mismatch")
        if topology(old) != topology(new):
            issues.append("topology_mismatch")
        for index in range(min(top, max(len(old), len(new)))):
            before = old[index] if index < len(old) else None
            after = new[index] if index < len(new) else None
            status = (
                "missing_baseline"
                if before is None
                else "missing_candidate" if after is None else "|".join(issues) or "comparable"
            )
            result = dict(
                scope="per_rank" if by_rank else "global",
                scope_rank=rank if by_rank else "-",
                tail_order=index + 1,
                status=status,
                baseline_rows=len(old),
                candidate_rows=len(new),
            )
            for side, row in (("baseline", before), ("candidate", after)):
                for field in identifiers + locations:
                    result[f"{side}_{field}"] = row[field] if row else "-"
            old_metrics = timeline_comparison_metrics(before) if before else {}
            new_metrics = timeline_comparison_metrics(after) if after else {}
            for field in TIMELINE_METRICS:
                old_value, new_value = old_metrics.get(field, "-"), new_metrics.get(field, "-")
                result[f"baseline_{field}"] = old_value
                result[f"candidate_{field}"] = new_value
                result[f"delta_{field}"] = (
                    new_value - old_value if status == "comparable" and old_value != "-" and new_value != "-" else "-"
                )
            report.append(result)
    return report


def report_timeline_comparison(report, baseline, candidate):
    print(
        "# Tail-order comparison: each run is independently sorted by descending observed_span_us; rows are NOT paired invocations."
    )
    print(
        "# All times are GPU-local us; delta=candidate-baseline. Negative observed-span delta means a shorter observed span, not proven kernel speedup."
    )
    print(
        "# comparable means equal per-rank sample counts and recorded topology; reports do not establish identical workloads, warmup phases, or hardware."
    )
    print(
        "# Signed gaps subtract per-stage maximum endpoints within one row; stages may refer to different peers/QPs. receive_minus_arrival is NOT same-peer processing time."
    )
    print(
        "# Producer includes inline posting; arrival is GPU marker observation. No cross-GPU clock subtraction or sum of overlapping spans."
    )
    print(
        "# Input times are rounded to 0.001 us. Locations retain the original report's representative event, not every tied QP."
    )
    print(
        "# Saved TSV cannot revalidate raw CSV completeness; retain original saved/dropped=0 checks. Mismatched samples/topology suppress deltas."
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=list(report[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in report:
        writer.writerow({field: f"{value:.3f}" if isinstance(value, float) else value for field, value in row.items()})
    for side, rows in (("baseline", baseline), ("candidate", candidate)):
        for rank in sorted({row["rank"] for row in rows}):
            local = [row for row in rows if row["rank"] == rank]
            selected = sum(row[f"{side}_rank"] == rank for row in report)
            arrivals = Counter(row["arrival_wait_location"] for row in local)
            frequencies = ";".join(f"{location}:{count}" for location, count in sorted(arrivals.items()))
            print(
                f"# {side} rank={rank} retained_rows={len(local)} selected_rows={selected} last_arrival_locations_all_rows={frequencies}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="CSV paths/globs, or one candidate TSV with --compare-timelines")
    parser.add_argument("--counts", action="store_true", help="summarize unsampled counts-only kind=13 records")
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="identify last observed dispatch stages; requires fresh kind=14/15 endpoint records",
    )
    parser.add_argument(
        "--compare-counts",
        action="append",
        metavar="OFF_GLOB",
        help="compare this OFF-run path/quoted glob with positional ON-run traces; repeat for multiple baseline inputs",
    )
    parser.add_argument(
        "--compare-timelines",
        metavar="BASELINE_TSV",
        help="compare slowest rows from two saved --timeline reports; positional input is the candidate TSV",
    )
    parser.add_argument("--top", type=int, help="with --compare-timelines, show top N per run/scope (default 5)")
    parser.add_argument(
        "--by-rank",
        action="store_true",
        help="with --compare-counts, aggregate ranks; with --compare-timelines, select top N separately per rank",
    )
    parser.add_argument(
        "--expected-ranks", type=int, help="with timeline or comparison modes, require ranks 0..N-1 in each run"
    )
    parser.add_argument(
        "--skip-invocations",
        type=int,
        default=0,
        help="skip first N local invocation ordinals, not automatic warmup detection",
    )
    args = parser.parse_args(argv)
    if args.compare_timelines is not None:
        try:
            if args.timeline or args.counts or args.compare_counts or len(args.traces) != 1:
                raise ValueError("--compare-timelines requires one candidate TSV and no other report mode")
            before_path, after_path = Path(args.compare_timelines).resolve(), Path(args.traces[0]).resolve()
            if before_path == after_path:
                raise ValueError("baseline and candidate contain the same file")
            baseline = load_timeline_report(before_path, args.skip_invocations, args.expected_ranks)
            candidate = load_timeline_report(after_path, args.skip_invocations, args.expected_ranks)
            report = compare_timeline_rows(baseline, candidate, args.top if args.top is not None else 5, args.by_rank)
        except (OSError, ValueError, csv.Error) as error:
            parser.error(str(error))
        print(f"# baseline_report={before_path} candidate_report={after_path}")
        report_timeline_comparison(report, baseline, candidate)
        return 0
    if args.top is not None:
        parser.error("--top requires --compare-timelines")
    if args.timeline:
        try:
            if args.counts or args.compare_counts or args.by_rank:
                raise ValueError("--timeline cannot be combined with count or comparison modes")
            if args.skip_invocations < 0 or (args.expected_ranks is not None and args.expected_ranks < 1):
                raise ValueError("skip count must be nonnegative and expected rank count must be positive")
            files, records, seen = {}, [], set()
            for pattern in args.traces:
                matches = [pattern] if Path(pattern).is_file() else sorted(glob(pattern))
                if not matches:
                    raise ValueError(f"no files match {pattern}")
                for match in matches:
                    path = Path(match).resolve()
                    if path in seen:
                        raise ValueError(f"duplicate input: {path}")
                    seen.add(path)
                    loaded = load_samples(path, timeline=True)
                    rank = loaded[0]["rank"]
                    if rank in files:
                        raise ValueError(f"multiple trace files for rank {rank}; select one run/context per rank")
                    files[rank] = str(path)
                    retained = [row for row in loaded if row["invocation"] > args.skip_invocations]
                    if not retained:
                        raise ValueError(f"{path}: no invocations remain after filtering")
                    records.extend(retained)
            if args.expected_ranks is not None and set(files) != set(range(args.expected_ranks)):
                raise ValueError(f"expected ranks 0..{args.expected_ranks-1}, found {sorted(files)}")
            report = dispatch_timeline(records)
        except (OSError, ValueError, csv.Error) as error:
            parser.error(str(error))
        report_timeline(report)
        return 0
    if args.compare_counts:
        try:
            if args.skip_invocations < 0 or (args.expected_ranks is not None and args.expected_ranks < 1):
                raise ValueError("skip count must be nonnegative and expected rank count must be positive")
            baseline, old_files = load_count_run(args.compare_counts, 0, args.skip_invocations, args.expected_ranks)
            candidate, new_files = load_count_run(args.traces, 1, args.skip_invocations, args.expected_ranks)
            if old_files.keys() != new_files.keys():
                raise ValueError(f"rank sets differ: baseline={sorted(old_files)}, candidate={sorted(new_files)}")
            if set(old_files.values()) & set(new_files.values()):
                raise ValueError("baseline and candidate contain the same file")
            report = compare_counts(baseline, candidate, args.by_rank)
        except (OSError, ValueError, csv.Error) as error:
            parser.error(str(error))
        report_comparison(report, args.by_rank)
        return 0
    if args.by_rank or args.expected_ranks is not None:
        parser.error("--by-rank requires a comparison mode; --expected-ranks requires a timeline or comparison mode")
    groups = defaultdict(list)
    count_records = []
    keys = ("file", "rank", "peer", "queue", "handler", "issued", "mode", "rows", "bytes")
    try:
        if args.skip_invocations < 0:
            raise ValueError("--skip-invocations must be nonnegative")
        seen = set()
        for pattern in args.traces:
            matches = [pattern] if Path(pattern).is_file() else sorted(glob(pattern))
            if not matches:
                raise ValueError(f"no files match {pattern}")
            for match in matches:
                path = Path(match).resolve()
                if path in seen:
                    raise ValueError(f"duplicate input: {path}")
                seen.add(path)
                for sample in load_samples(path, counts_only=args.counts):
                    if sample["invocation"] > args.skip_invocations:
                        if args.counts:
                            count_records.append(sample)
                        else:
                            groups[tuple(sample[field] for field in keys)].append(sample)
        if not groups and not count_records:
            raise ValueError("no sampled submissions remain after filtering")
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    if args.counts:
        report_counts(count_records)
        return 0
    print("# ns p50/p95: sampled calls, not kernel latency; files/ranks/QPs/handler/issued status stay separate.")
    print("# handler=2: GPU_SM_DB; 10: GPU_SM_NO_DBR; other raw values follow AUTO proxy fallback.")
    print(
        "# lock_held overlaps index/DB/DBREC stages; other excludes lock_wait+lock_held for DB, index/fence+first for others."
    )
    print(
        "# '-' means stage does not apply; issued=0 is a covered/skipped DB; proxy issued=1 is a request, not NIC progress."
    )
    print(
        "# Timing inside the lock can amplify contention. Verify each rank's EPTIMING saved/dropped=0 log; CSV has no footer."
    )
    fields = [
        *keys,
        "samples",
        *[f"{metric}_{percentile_name}" for metric in METRICS for percentile_name in ("p50", "p95")],
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for key, samples in sorted(groups.items()):
        result = dict(zip(keys, key))
        result["samples"] = len(samples)
        if len(samples) < 20:
            print(f"WARNING: only {len(samples)} samples for {key}; sparse p95", file=sys.stderr)
        for metric in METRICS:
            values = [sample[metric] for sample in samples if sample[metric] is not None]
            for name, fraction in (("p50", 0.5), ("p95", 0.95)):
                result[f"{metric}_{name}"] = f"{percentile(values, fraction):.1f}" if values else "-"
        writer.writerow(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
