import pandas as pd
import pathlib
import requests
import json

ABS_PATH = pathlib.Path(__file__).parent.absolute()


dictionary = {}
r_dictionary = {}


class Dataset(object):
    def __init__(self):
        self.folder = f"{ABS_PATH}/dataset"
        self.file = f"{self.folder}/LeagueofLegends.csv"
        self.dataset = None

    def get_data(self):
        required_columns = [
            'blueTopChamp',
            'blueJungleChamp',
            'blueMiddleChamp',
            'blueADCChamp',
            'blueSupportChamp',
            'redTopChamp',
            'redJungleChamp',
            'redMiddleChamp',
            'redADCChamp',
            'redSupportChamp',
            'rResult']
        dataset = pd.read_csv(self.file, usecols=required_columns)
        dataset = dataset.rename(columns={'rResult': 'redWin'})
        self.dataset = dataset
        self.create_dictionary_names()
        return dataset

    def get_training_data(self):
        x = self.dataset.loc[:, "blueTopChamp":"redSupportChamp"]
        y = self.dataset.loc[:, "redWin"]
        return x, y

    def show_dataset(self):
        if self.dataset is None:
            print("no dataset loaded")
        else:
            print(self.dataset)

    def create_dictionary_names(self):
        i = 0
        for key in self.dataset.loc[:, "blueTopChamp":"redSupportChamp"]:
            for value in self.dataset[key].values:
                if value not in dictionary.keys():
                    dictionary[value] = i
                    r_dictionary[i] = value
                    i += 1

    def _replace_names_with_key(self):
        self.dataset.replace(to_replace=dict(dictionary), inplace=True)

    def _replace_keys_with_names(self):
        cols = ['blueTopChamp',
                'blueJungleChamp',
                'blueMiddleChamp',
                'blueADCChamp',
                'blueSupportChamp',
                'redTopChamp',
                'redJungleChamp',
                'redMiddleChamp',
                'redADCChamp',
                'redSupportChamp']
        for col in cols:
            self.dataset.replace({col: r_dictionary}, inplace=True)

    def human_data(self):
        self._replace_keys_with_names()

    def prepare_data(self):
        self._replace_names_with_key()


if __name__ == "__main__":
    db = Dataset()
    db.get_data()
    db.show_dataset()
    db.prepare_data()
    db.show_dataset()
