from .checker.version import CheckVersion

def Run():
    preRun()

def preRun():
    CheckVersion("3.7")
