import os
import shutil
import clang.cindex
from mscclpp.libclang_utils import find_kernel, build_str

def get_func_decl_data(kernel: clang.cindex.Cursor, skip_rank: bool) -> (list[str], list[str]):
    new_params = list()
    call_args = list()
    if skip_rank == True:
        for arg in kernel.get_arguments():
            new_params.append(arg.type.spelling + " " + arg.displayname)
            if arg.displayname == "rank" or arg.type.spelling != "int":
                # new_params.append(arg.type.spelling + " " + arg.displayname)
                call_args.append(arg.displayname)
    else:
        for arg in kernel.get_arguments():
            new_params.append(arg.type.spelling + " " + arg.displayname)
            if arg.type.spelling != "int":
                # new_params.append(arg.type.spelling + " " + arg.displayname)
                call_args.append(arg.displayname)
    return (new_params, call_args)

def code_gen(file_dir: str, file_name: str, kernel_name: str, list_args: list[int], ) -> (str, str):
    if(len(list_args) < 2):
        raise Exception('Not enough arguments!')
    
    idx = clang.cindex.Index.create()
    input_file_path = file_dir +"/"+ file_name
    tu = idx.parse(input_file_path, args=[f'-I{file_dir}/../../include'])
    kernel = find_kernel(tu.cursor, kernel_name)
    if kernel is None:
        raise Exception('Kernel not found!')
    
    new_params, call_args = get_func_decl_data(kernel, True)
    for args in list_args:
        call_args.append(args)

    new_params_str = build_str(new_params, ', ')
    new_kernel_name = kernel.spelling + "_new"
    kernel_prefix = "extern \"C\" __global__ void LAUNCH_BOUNDS"
    func_decl = build_str([kernel_prefix, new_kernel_name, "(",new_params_str,")"], " ")

    call_args_str = build_str(call_args[:-1], ', ')
    func_body = ""    
    if kernel_name == "allreduce1":
        read_only = str(call_args[-1])
        func_body = f'''{{
    {kernel.spelling}_helper <{read_only}> ({call_args_str});
}}'''
    elif kernel_name == "alreduce4":
        num_threads = str(call_args[-1])
        num_blocks = str(call_args[-3])
        func_body = f'''{{
    {kernel.spelling}<<{num_blocks}, {num_threads}>>({call_args_str});
}}'''
    else:
        num_threads = str(call_args[-1])
        num_blocks = str(call_args[-2])
        func_body = f'''{{
    {kernel.spelling}<<{num_blocks}, {num_threads}>>({call_args_str});
}}'''
    output_file_name = kernel_name + "_new.cu"
    output_file_path=file_dir+"/"+output_file_name
    with open(input_file_path, 'r') as file:
        new_kernel = func_decl + func_body
        new_file_content = file.read() + '\n' + new_kernel
        with open(output_file_path, 'w') as new_file:
            new_file.write(new_file_content)

    
    return output_file_name, new_kernel_name