from mscclpp.language.internal.globals import get_program


def JSON():
    get_program().post_process_operations()
    return get_program().to_json()
