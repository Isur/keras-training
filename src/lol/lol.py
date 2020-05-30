from .dataset import Dataset


class Lol(object):
    def __init__(self):
        self.dataset = None

    def load_dataset(self):
        if self.dataset is not None:
            print("dataset already loaded!")
            return
        db = Dataset()
        self.dataset = db.get_data()
        print("loaded dataset:")
        db.show_dataset()


def main():
    lol = Lol()
    lol.load_dataset()
