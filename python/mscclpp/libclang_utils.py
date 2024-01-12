import clang.cindex

def find_kernel(node: clang.cindex.Cursor, kernel_name: str ) -> clang.cindex.Cursor:
    if node.kind == clang.cindex.CursorKind.FUNCTION_DECL and node.is_definition():
        if kernel_name == node.spelling:
            return node
    for n in node.get_children():
        kernel = find_kernel(n, kernel_name)
        if kernel is not None:
            return kernel

def code_gen_from_cursor(cursor: clang.cindex.Cursor) -> list[str]:
    code = [] 
    line = ""
    prev_token = None
    for curr_token in cursor.get_tokens():
        if prev_token is None:
            prev_token = curr_token
        prev_location = prev_token.location
        prev_token_end_col = prev_location.column + len(prev_token.spelling)
        cur_location = curr_token.location
        if cur_location.line > prev_location.line:
            code.append(line)
            line = " " * (cur_location.column - 1)
        else:
            if cur_location.column > (prev_token_end_col):
                line += " "
        line += curr_token.spelling
        prev_token = curr_token
    if len(line.strip()) > 0:
        code.append(line)
    del line, prev_token, prev_location, prev_token_end_col, cur_location, line
    return code

def dump_children(node: clang.cindex.Cursor):
    for n in node.get_children():
        print (n.kind, n.type.spelling, n.displayname)
        dump_children(n)

def get_func_decl_data(kernel: clang.cindex.Cursor, skip_rank: bool) -> (list[str], list[str]):
    new_params = list()
    call_args = list()
    if skip_rank == True:
        for arg in kernel.get_arguments():
            if arg.displayname == "rank" or arg.type.spelling != "int":
                new_params.append(arg.type.spelling + " " + arg.displayname)
                call_args.append(arg.displayname)
    else:
        for arg in kernel.get_arguments():
            if arg.type.spelling != "int":
                new_params.append(arg.type.spelling + " " + arg.displayname)
                call_args.append(arg.displayname)
    return (new_params, call_args)

def build_str(args: list[str], sep: str) -> str:
    args_str = sep.join(str(v) for v in args)
    return args_str
