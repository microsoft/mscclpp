# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from python.mscclpp.language.program import CollectiveProgram


def compile(
    algo,
    name: str,
    collective: str,
    nranks_per_node: int,
    world_size: int,
    instances: int,
    protocol: str,
    num_threads_per_block: int = 1024,
    min_msg_size: int = 0,
    max_msg_size: int = 2**64 - 1,
    tags: dict = {},
    **kwargs,
):
    """Compile a MSCCL++ program from a high-level algorithm description.
    Args:
        algo: The high-level algorithm description (e.g., a function or class).
        name (str): The name of the program.
        collective (str): The collective operation type (e.g., "allreduce").
        nranks_per_node (int): Number of ranks per node.
        world_size (int): Total number of ranks in the program.
        instances (int): Number of instances to replicate.
        protocol (str): Communication protocol ("Simple" or "LL").
        num_threads_per_block (int): Number of threads per GPU thread block.
        min_msg_size (int): Minimum message size for this program.
        max_msg_size (int): Maximum message size for this program.
        tags (dict): Additional tags or metadata for the program.
        **kwargs: Additional keyword arguments for future extensions.
    Returns:
        The compiled program object.
    Raises:
        NotImplementedError: If the compilation logic is not implemented.
    """
    prog: CollectiveProgram = algo(
        name,
        collective,
        nranks_per_node,
        world_size,
        instances,
        protocol,
        num_threads_per_block,
        min_msg_size,
        max_msg_size,
        **kwargs,
    )
    prog.to_json()
    pass
