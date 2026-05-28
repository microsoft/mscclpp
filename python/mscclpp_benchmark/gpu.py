# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

_API_NAMES = {
    "get_device_count": ("hipGetDeviceCount", "cudaGetDeviceCount"),
    "get_device": ("hipGetDevice", "cudaGetDevice"),
    "get_device_properties": ("hipGetDeviceProperties", "cudaGetDeviceProperties"),
    "set_device": ("hipSetDevice", "cudaSetDevice"),
    "stream_begin_capture": ("hipStreamBeginCapture", "cudaStreamBeginCapture"),
    "stream_end_capture": ("hipStreamEndCapture", "cudaStreamEndCapture"),
    "graph_instantiate": ("hipGraphInstantiate", "cudaGraphInstantiate"),
    "graph_launch": ("hipGraphLaunch", "cudaGraphLaunch"),
    "graph_destroy": ("hipGraphDestroy", "cudaGraphDestroy"),
    "graph_exec_destroy": ("hipGraphExecDestroy", "cudaGraphExecDestroy"),
    "get_error_string": ("hipGetErrorString", "cudaGetErrorString"),
}


@dataclass(frozen=True)
class _Runtime:
    name: str
    success: Any
    capture_mode_relaxed: Any
    funcs: dict[str, Callable[..., Any] | None]

    @classmethod
    def create(cls, name: str, module: Any, success: Any, capture_mode_relaxed: Any) -> "_Runtime":
        index = 0 if name == "hip" else 1
        funcs = {
            attr: (None if names[index] is None else getattr(module, names[index]))
            for attr, names in _API_NAMES.items()
        }
        return cls(name=name, success=success, capture_mode_relaxed=capture_mode_relaxed, funcs=funcs)

    def call(self, name: str, *args: Any) -> tuple[Any, ...]:
        fn = self.funcs[name]
        if fn is None:
            raise RuntimeError(f"{name} is not available for {self.name}")
        result = fn(*args)
        if not isinstance(result, tuple):
            result = (result,)
        self.check(result[0], name)
        return result[1:]

    def check(self, error: Any, api: str) -> None:
        if error == self.success:
            return
        result = self.funcs["get_error_string"](error)
        if not isinstance(result, tuple):
            result = (result,)
        err, message = result
        if err != self.success:
            raise RuntimeError(f"{api} failed with error {int(error)}")
        decoded = message.decode("utf-8") if isinstance(message, bytes) else str(message)
        raise RuntimeError(f"{api} failed: {decoded} ({int(error)})")


def _load_runtime() -> _Runtime:
    errors: list[str] = []

    try:
        from hip import hip

        runtime = _Runtime.create(
            name="hip",
            module=hip,
            success=hip.hipError_t.hipSuccess,
            capture_mode_relaxed=hip.hipStreamCaptureMode.hipStreamCaptureModeRelaxed,
        )
        count = runtime.call("get_device_count")[0]
        if count and count > 0:
            return runtime
        errors.append(f"hipGetDeviceCount returned count={count}")
    except ImportError as exc:
        errors.append(f"hip-python unavailable: {exc}")

    try:
        from cuda.bindings import runtime as cuda_runtime

        runtime = _Runtime.create(
            name="cuda",
            module=cuda_runtime,
            success=cuda_runtime.cudaError_t.cudaSuccess,
            capture_mode_relaxed=cuda_runtime.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed,
        )
        count = runtime.call("get_device_count")[0]
        if count and count > 0:
            return runtime
        errors.append(f"cudaGetDeviceCount returned count={count}")
    except ImportError as exc:
        errors.append(f"cuda-bindings unavailable: {exc}")

    raise RuntimeError("No usable CUDA/HIP Python runtime found: " + "; ".join(errors))


_RUNTIME = _load_runtime()


class Graph:
    def __init__(self, graph_exec: Any) -> None:
        self._graph_exec = graph_exec

    def launch(self, stream: Any) -> None:
        _api("graph_launch")(self._graph_exec, _stream_ptr(stream))

    def close(self) -> None:
        if self._graph_exec is not None:
            _api("graph_exec_destroy")(self._graph_exec)
            self._graph_exec = None


def init_runtime() -> None:
    return None


def capture_graph(stream: Any, capture_fn: Callable[[], None]) -> Graph:
    _api("set_device")(current_device())
    stream_ptr = _stream_ptr(stream)
    _api("stream_begin_capture")(stream_ptr, _RUNTIME.capture_mode_relaxed)

    graph = None
    try:
        capture_fn()
        graph = _api("stream_end_capture")(stream_ptr)[0]
    except Exception:
        try:
            _api("stream_end_capture")(stream_ptr)
        except Exception:
            pass
        raise

    try:
        graph_exec = _instantiate_graph(graph)
        return Graph(graph_exec)
    finally:
        if graph is not None:
            _api("graph_destroy")(graph)


def current_device() -> int:
    return int(_api("get_device")()[0])


def device_name(device_id: int | None = None) -> str:
    if device_id is None:
        device_id = current_device()
    prop = _api("get_device_properties")(int(device_id))[0]
    name = getattr(prop, "name", "UNKNOWN")
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def _stream_ptr(stream: Any) -> int:
    return int(getattr(stream, "ptr", stream))


def _instantiate_graph(graph: Any) -> Any:
    if _RUNTIME.name == "hip":
        return _api("graph_instantiate")(graph, None, 0)[0]
    return _api("graph_instantiate")(graph, 0)[0]


def _api(name: str) -> Callable[..., tuple[Any, ...]]:
    api = globals().get(name)
    if api is None:
        api = __getattr__(name)
    return api


def _make_api(name: str) -> Callable[..., tuple[Any, ...]]:
    def api(*args: Any) -> tuple[Any, ...]:
        return _RUNTIME.call(name, *args)

    api.__name__ = name
    return api


def __getattr__(name: str) -> Callable[..., tuple[Any, ...]]:
    if name in _API_NAMES:
        api = _make_api(name)
        globals()[name] = api
        return api
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
