_current_program = None

def set_curr(program):
    global _current_program
    _current_program = program

def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program