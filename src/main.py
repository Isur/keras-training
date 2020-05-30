from .checker.version import CheckVersion
from .lol.lol import main as lol


def Run():
    preRun()
    lol()


def preRun():
    CheckVersion("3.7")
