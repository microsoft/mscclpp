Integrate Torch with two approaches:
1. Using our nccl API directly. Without any modifications to Torch.
2. Modifying Torch to use mscclpp as a backend instead of nccl.

Both approaches support customized communication collectives.
There are two ways to define custom collectives:
1. Using cuda kernel and msccl++ cpp API.
2. Using msccl++ python DSL.

For algorithm selection, user can provide a selector to select algorithms based on
various parameters such as tensor size, data type, device topology, etc.

A selector can be registered as a c++ function or a python function. Only one selector can be registered. If call register_selector multiple times, the last registered selector will be used.

A pic for this high-level design is shown below:

```{figure} ../figs/customize_algo.png
:name: MSCCL++ Customization Algorithm Selection
:alt: MSCCL++ Customization Algorithm Selection
:align: center
:width: 400px

MSCCL++ Customization Algorithm Selection Overview
```

## The algorithm definition
```python
class Algorithm:
    name: str
    collective: mscclpp.Collective
    plan_handle: mscclpp.PlanHandle
    native_handle # The handle for cpp algorithm
    min_size: int
    max_size: int
    architectures: List[str]
    buffer_mode: mscclpp.BufferMode # in-place/out-of-place/all
    constraint: mscclpp.AlgorithmConstraint
    tags: Dict[str, str]



def execute(comm, rank, nblocks, nthreads_per_block, input, output, count, dtype, extra, stream):
    if is_dsl_based():
        # will ignore nblocks as it is build-in in the DSL
        self.executor.execute()
        pass
    elif is_kernel_based():
        self.native_handle.launch()
        pass

def is_dsl_based() -> bool:
    pass

def is_kernel_based() -> bool:
    pass

def clean_cached_contexts():
    # clear all cached contexts
    pass
```

## Workflow in user application

Get the algorithm via MSCCL++ DSL
```python
import mscclpp
collective_program = your_dsl_program_definition()
plan_handle = mscclpp.compile(algo=allreduce_nvls, algo_spec=spec, rank=rank)
algorithm = mscclpp.Algorithm.from_execution_plan(spec=spec, plan_handle=plan_handle ...)

# if you want to share the algorithm with nccl API
mscclpp.register_algorithm(algorithm)
```

Get the algorithm via MSCCL++ C++ API
```cpp
// define the algorithm by implementing AlgorithmBuilder
class AllreduceAlgoBuilder : public mscclpp::AlgorithmBuilder {
    mscclpp::Algorithm build() override {
        // build the algorithm using mscclpp C++ API
    }
};
```
Build the algorithm in python code
```python
native_handle = mscclpp.compile(file="allreduce_kernel_based_algo.cu") # jit compilation
algorithm = mscclpp.Algorithm.from_native_handle(native_handle=native_handle, ...)

# if you want to share the algorithm with nccl API
mscclpp.register_algorithm(algorithm)
```

## Register the selector
If you want to use NCCL API with cutomized selector. You need to register your selector explicitly.


## Tunning the algorithm
```python
# register all algorithms
dsl_algo_configs_to_try = {"allreduce2nodes": {"ninstances": [1, 2, 4]}}
for algo_name, algo_config in dsl_algo_configs_to_try.items():
    spec = mscclpp.ExecutionPlanSpec(..dsl_algo_configs_to_try)
    plan_handle = mscclpp.compile(algo_name=algo_name, algo_spec=spec)
    algorithm = mscclpp.Algorithm.from_execution_plan(spec=spec, plan_handle=plan_handle)
    algo_collection.add_algorithm(algorithm)

mscclpp_native_handle = mscclpp.compile(file="allreduce_kernel_based_algo.cu")
for native_handle in mscclpp_native_handle.handles:
    algorithm = mscclpp.Algorithm.from_native_handle(native_handle=native_handle, ...)
    algo_collection.add_algorithm(algorithm)

algo_collection = mscclpp.AlgorithmCollection(collective=mscclpp.Collective.ALLREDUCE)
message_sizes = [1024, 2048, 4096, 8192]
nblocks_to_try = [16, 32, 64, 128]
nthreads_per_block_to_try = [128, 256, 512, 1024]

best_config = None
for msg_size in message_sizes:
    for algorithm in algo_collection.get_algorithms():
        if not (algorithm.min_size <= msg_size <= algorithm.max_size):
            continue
        for nblocks in nblocks_to_try:
            for nthreads_per_block in nthreads_per_block_to_try:
                time = benchmark(algorithm, nblocks, nthreads_per_block, msg_size)
                log_result(algorithm, nblocks, nthreads_per_block, msg_size, time)
                if best_config is None or time < best_config['time']:
                    best_config = {
                        'algorithm': algorithm,
                        'nblocks': nblocks,
                        'nthreads_per_block': nthreads_per_block,
                        'msg_size': msg_size,
                        'time': time
                    }
# clear all contexts after tuning
for algo in algo_collection:
    algo.clean_cached_contexts()
```