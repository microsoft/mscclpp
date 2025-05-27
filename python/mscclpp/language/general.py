from mscclpp.language.internal.globals import get_program
import json


def JSON():
    return get_program().to_json()
