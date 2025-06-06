from mscclpp.language.internal.globals import get_program


def JSON():
    get_program().optimize_operations()
    return get_program().to_json()
