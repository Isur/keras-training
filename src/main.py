from .checker.version import CheckVersion
from .wine import wine


def Run():
    preRun()
    wine.Wine()


def preRun():
    CheckVersion("3.7")
