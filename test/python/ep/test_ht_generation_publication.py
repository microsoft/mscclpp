from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_ht_host_visibility_is_generation_qualified():
    runtime = (ROOT / "src/ext/ep/ht_runtime.cc").read_text()
    kernel = (ROOT / "src/ext/ep/high-throughput/dispatch.cu").read_text()
    config = (ROOT / "src/ext/ep/high-throughput/config.cuh").read_text()
    assert "DispatchCountPublication" in config
    assert "alignas(64)" in config
    assert "uint64_t generation" in config
    assert "expectedGeneration" in kernel
    assert "memoryOrderRelease" in kernel and "scopeSystem" in kernel
    assert "__atomic_load_n" in runtime and "__ATOMIC_ACQUIRE" in runtime
    assert "observedGeneration == expectedGeneration" in runtime
    assert "observedGeneration > expectedGeneration" in runtime
    assert "expected generation" in runtime and "observed generation" in runtime
    assert "notifyPoisoned_ = true" in runtime
    assert "moeRecvCounter_ = -1" not in runtime
    assert "moeRecvExpertCounter_[i] = -1" not in runtime


def test_generation_zero_reserved_and_single_inflight():
    runtime = (ROOT / "src/ext/ep/ht_runtime.cc").read_text()
    header = (ROOT / "src/ext/ep/ht_runtime.hpp").read_text()
    assert "uint64_t notifyGeneration_ = 0" in header
    assert "bool notifyInFlight_ = false" in header
    assert "const uint64_t expectedGeneration = ++notifyGeneration_" in runtime
    assert "generation zero is reserved" in runtime
    assert "one in-flight notify per runtime" in runtime
    assert "collective quiescent reinit required" in runtime


def test_ht_phases_use_dedicated_semaphore_sets():
    runtime = (ROOT / "src/ext/ep/ht_runtime.cc").read_text()
    header = (ROOT / "src/ext/ep/ht_runtime.hpp").read_text()
    for phase in ("count", "dispatch", "combine"):
        assert f"{phase}BarrierChannels_" in header
        assert f"{phase}BarrierChannelHandles_" in header
    assert "CountBarrierConnectionTag = 19" in runtime
    assert "DispatchBarrierConnectionTag = 20" in runtime
    assert "CombineBarrierConnectionTag = 21" in runtime
    assert "countBarrierChannelHandles_.get(), rank_, stream, numChannels" in runtime
    assert "dispatchBarrierChannelHandles_.get(), rank_" in runtime
    assert "combineBarrierChannelHandles_.get(), rank_" in runtime
    assert "does not advance count-exchange semaphore phases" in runtime
    assert "barrierChannelHandles_" not in runtime


def test_single_gpu_thread_publishes_payload_then_generation():
    kernel = (ROOT / "src/ext/ep/high-throughput/dispatch.cu").read_text()
    elected = kernel[kernel.index("if (threadId == 0)") :]
    payload = elected.index("mappedPublication->numRecvTokens =")
    experts = elected.index("mappedPublication->numRecvTokensPerExpert")
    generation = elected.index("&mappedPublication->generation")
    assert payload < experts < generation
