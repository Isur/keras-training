from .checker.version import CheckVersion
from .keras.keras import Keras


def Run():
    preRun()
    Keras()


def preRun():
    CheckVersion("3.7")
