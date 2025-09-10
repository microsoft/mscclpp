import argparse
from mscclpp.language.channel import *
from mscclpp.language.rank import *
from mscclpp.language.general import *
from mscclpp.language.program import *
from mscclpp.language.collectives import *


def alltoallv_variable_example(name, gpu_size, num_threads_per_block, min_message_size, max_message_size):
    """
    AllToAllV with placeholder variables for runtime chunk size determination
    This creates a template that can be instantiated with actual sizes at runtime
    """
    blockSize = 32768
    maxThreadBlocks = 32
    # TODO: Support new AllToAllv in DSL
    collective = AllToAllv(gpu_size, blockSize, maxThreadBlocks, True)
    
    with CollectiveProgram(
        name,
        collective,
        gpu_size,
        instances=1,
        protocol="Simple",  # Use valid protocol - we'll add dynamic behavior in post-processing
        num_threads_per_block=num_threads_per_block,
        use_double_scratch_buffer=False,
        min_message_size=min_message_size,
        max_message_size=max_message_size,
    ):
        # Create channels and buffers
        channels = {}
        scratch_buffer = {}
        
        for gpu in range(gpu_size):
            src_rank_id = gpu
            # Use maximum possible scratch buffer size for template
            scratch_buffer[src_rank_id] = VariableBuffer(src_rank_id, gpu_size - 1)
            
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != dst_rank_id:
                    channels[dst_rank_id, src_rank_id] = MemoryChannel(dst_rank_id, src_rank_id)

        # Phase 1: Put data to remote scratch buffers
        for gpu in range(gpu_size):
            src_rank_id = gpu
            src_rank = Rank(src_rank_id)
            input_buffer = src_rank.get_variable_input_buffer()
            
            # First, handle local copy for same rank data
            output_buffer = src_rank.get_variable_output_buffer()
            src_rank.copy(
                output_buffer[src_rank_id : src_rank_id + 1],
                input_buffer[src_rank_id : src_rank_id + 1],
                dynamic_tbgroup_id=0,
            )
            
            # Then send data to other ranks
            for peer in range(gpu_size):
                dst_rank_id = peer
                if dst_rank_id != src_rank_id:
                    tb = dst_rank_id if dst_rank_id < src_rank_id else dst_rank_id - 1
                    
                    # Use concrete chunk indices - the dynamic system will modify these
                    remote_index = src_rank_id if src_rank_id < dst_rank_id else src_rank_id - 1
                    channels[dst_rank_id, src_rank_id].put(
                        scratch_buffer[dst_rank_id][remote_index : remote_index + 1],
                        input_buffer[dst_rank_id : dst_rank_id + 1],
                        dynamic_tbgroup_id=tb,
                    )
                    # TODO: support dynamic_tbgroup_id in template json, assign signal to thread block id 0 of each thread block group (tbgroup)
                    channels[dst_rank_id, src_rank_id].signal(dynamic_tbgroup_id=tb, data_sync=SyncType.before)

        # Phase 2: Each rank receives data from its scratch buffer
        for gpu in range(gpu_size):
            dst_rank_id = gpu
            dst_rank = Rank(dst_rank_id)
            output_buffer = dst_rank.get_output_buffer()
            
            # Receive data from all other ranks
            for peer in range(gpu_size):
                src_rank_id = peer
                if src_rank_id != dst_rank_id:
                    # Calculate the index in the scratch buffer where this rank's data is stored
                    index = src_rank_id if src_rank_id < dst_rank_id else src_rank_id - 1
                    tb = index
                    
                    # Wait for data to arrive from the source rank
                    # TODO: Verify if the order of dst_rank_id, src_rank_id is correct
                    # TODO: support dynamic_tbgroup_id in template json, assign wait to thread block id 0 of each thread block group (tbgroup)
                    channels[dst_rank_id, src_rank_id].wait(dynamic_tbgroup_id=tb, data_sync=SyncType.after)
                    
                    # Copy from local scratch buffer to output buffer
                    # Both buffers are on the same rank (dst_rank_id)
                    dst_rank.copy(
                        output_buffer[src_rank_id : src_rank_id + 1],
                        scratch_buffer[dst_rank_id][index : index + 1],
                        dynamic_tbgroup_id=tb,
                    )

        # Get the JSON and modify it to add dynamic support
        json_plan = JSON()
        
        # Add dynamic metadata to the generated JSON
        import json
        plan_dict = json.loads(str(json_plan))
        
        # Mark as dynamic and add template parameters
        plan_dict["dynamic"] = True
        plan_dict["dynamic_parameters"] = {
            "max_thread_blocks": "32",
            "block_size": "32768"
        }

        # Modify each GPU to use dynamic chunk variables
        for gpu_data in plan_dict["gpus"]:
            gpu_data.pop("input_chunks")
            gpu_data.pop("output_chunks")
            gpu_data.pop("scratch_chunks")

            gpu_data["dynamic_input_chunks"] = gpu_size
            gpu_data["dynamic_output_chunks"] = gpu_size
            gpu_data["dynamic_scratch_chunks"] = gpu_size - 1
            
            # Add operation templates for runtime instantiation
            if "threadblocks" in gpu_data:
                i = 0
                for tb in gpu_data["threadblocks"]:
                    tb["dynamic_tbgroup_id"] = i
                    i += 1
                    # Mark operations as templates that need runtime instantiation
                    if "ops" in tb:
                        for op in tb["ops"]:
                            if "src_buff" in op or "dst_buff" in op:
                                # For buffer references, add dynamic chunk mapping
                                if "src_buff" in op:
                                    for buff in op["src_buff"]:
                                        buff["dynamic_index"] = 0
                                        buff["dynamic_size"] = 1
                                        buff.pop("index")
                                        buff.pop("size")
                                        
                                if "dst_buff" in op:
                                    for buff in op["dst_buff"]:
                                        buff["dynamic_index"] = 0
                                        buff["dynamic_size"] = 1
                                        buff.pop("index")
                                        buff.pop("size")
                    tb.pop("id")
        
        # Output the modified JSON
        print(json.dumps(plan_dict, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--num_threads_per_block", type=int, default=1024)
    parser.add_argument("--min_message_size", type=int, default=1024)
    parser.add_argument("--max_message_size", type=int, default=1048576)
    
    args = parser.parse_args()
    alltoallv_variable_example(args.name, args.num_gpus, args.num_threads_per_block, 
                              args.min_message_size, args.max_message_size)