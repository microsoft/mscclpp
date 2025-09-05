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
    chunksperloop = 1
    collective = AllToAll(gpu_size, chunksperloop, True)
    
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
            scratch_buffer[src_rank_id] = Buffer(src_rank_id, gpu_size - 1)
            
            for peer in range(gpu_size):
                dst_rank_id = peer
                if src_rank_id != dst_rank_id:
                    channels[dst_rank_id, src_rank_id] = MemoryChannel(dst_rank_id, src_rank_id)

        # Phase 1: Put data to remote scratch buffers
        for gpu in range(gpu_size):
            src_rank_id = gpu
            src_rank = Rank(src_rank_id)
            input_buffer = src_rank.get_input_buffer()
            
            # First, handle local copy for same rank data
            output_buffer = src_rank.get_output_buffer()
            src_rank.copy(
                output_buffer[src_rank_id : src_rank_id + 1],
                input_buffer[src_rank_id : src_rank_id + 1],
                tb=0,
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
                        tb=tb,
                    )
                    channels[dst_rank_id, src_rank_id].signal(tb=tb, data_sync=SyncType.before)

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
                    channels[dst_rank_id, src_rank_id].wait(tb=tb, data_sync=SyncType.after)
                    
                    # Copy from local scratch buffer to output buffer
                    # Both buffers are on the same rank (dst_rank_id)
                    dst_rank.copy(
                        output_buffer[src_rank_id : src_rank_id + 1],
                        scratch_buffer[dst_rank_id][index : index + 1],
                        tb=tb,
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
            gpu_data["input_chunks"] = "${DYNAMIC_INPUT_CHUNKS}"
            gpu_data["output_chunks"] = "${DYNAMIC_OUTPUT_CHUNKS}"
            gpu_data["scratch_chunks"] = "${DYNAMIC_SCRATCH_CHUNKS}"
            
            # Add operation templates for runtime instantiation
            if "threadblocks" in gpu_data:
                for tb in gpu_data["threadblocks"]:
                    # Mark operations as templates that need runtime instantiation
                    if "ops" in tb:
                        for op in tb["ops"]:
                            if "src_buff" in op or "dst_buff" in op:
                                op["template"] = True
                                op["dynamic_size"] = "${chunk_size}"
                                op["dynamic_step"] = "${step_id}"
                                
                                # Add additional template variables for comprehensive dynamic support
                                op["dynamic_input_chunk"] = "${chunk_id}"
                                op["dynamic_output_chunk"] = "${chunk_id}"
                                op["dynamic_peer"] = "${peer_rank}"
                                op["dynamic_threadblock_count"] = "${tb_count}"
                                
                                # For buffer references, add dynamic chunk mapping
                                if "src_buff" in op:
                                    for buff in op["src_buff"]:
                                        buff["dynamic_index"] = "${src_chunk_index}"
                                        buff["dynamic_size"] = "${src_chunk_size}"
                                        
                                if "dst_buff" in op:
                                    for buff in op["dst_buff"]:
                                        buff["dynamic_index"] = "${dst_chunk_index}"
                                        buff["dynamic_size"] = "${dst_chunk_size}"
            
            # Also add operation templates at the GPU level (for compatibility with different JSON structures)
            if "operations" not in gpu_data:
                gpu_data["operations"] = []
            
            # Add a comprehensive operation template
            operation_template = {
                "operation_template": {
                    "type": "${operation_type}",  # put, get, copy, etc.
                    "inputChunk": "${chunk_id}",
                    "outputChunk": "${chunk_id}",
                    "peer": "${peer_rank}",
                    "channel": "${channel_id}",
                    "threadblock_count": "${tb_count}",
                    "size": "${chunk_size}",
                    "step": "${step_id}",
                    "src_buff": [
                        {
                            "type": "${src_buffer_type}",
                            "index": "${src_chunk_index}",
                            "size": "${src_chunk_size}"
                        }
                    ],
                    "dst_buff": [
                        {
                            "type": "${dst_buffer_type}",
                            "index": "${dst_chunk_index}",
                            "size": "${dst_chunk_size}"
                        }
                    ]
                }
            }
            gpu_data["operations"].append(operation_template)
        
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