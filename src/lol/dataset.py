import pandas as pd
import pathlib


ABS_PATH = pathlib.Path(__file__).parent.absolute()


class Dataset(object):
    def __init__(self):
        self.folder = f"{ABS_PATH}/dataset"
        self.file = f"{self.folder}/LeagueofLegends.csv"
        self.dataset = None

    def get_data(self, columns=None):
        if columns is None:
            required_columns = [
                'redJungleChamp',
                'redTopChamp',
                'redMiddleChamp',
                'redADCChamp',
                'redSupportChamp',
                'blueJungleChamp',
                'blueTopChamp',
                'blueMiddleChamp',
                'blueADCChamp',
                'blueSupportChamp',
                'rResult']
        else:
            required_columns = columns

        dataset = pd.read_csv(self.file, usecols=required_columns)
        dataset = dataset.rename(columns={'rResult': 'redWin'})
        self.dataset = dataset
        return dataset

    def show_dataset(self):
        if self.dataset is None:
            print("no dataset loaded")
        else:
            print(self.dataset)


if __name__ == "__main__":
    db = Dataset()
    db.get_data()
    db.show_dataset()
