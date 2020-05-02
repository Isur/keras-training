import sys


def CheckVersion(required):
    ver = sys.version
    if not ver.startswith(required):
        raise Exception(f"Wrong version of python.\n\
                        Required: {required}\n Used: {ver}")
