from src.SharedDataFrame import SharedDataFrame


class DataSets:
    """ Class to keep track of all uploaded datasets """
    def __init__(self):
        self.datasets = {}

    def add_dataset(self, filename, sdf: SharedDataFrame):
        self.datasets.update({filename: sdf})

    def get_datasets(self):
        return self.datasets

    def get_dataset(self, filename):
        return self.datasets.get(filename)

    def get_names(self) -> list:
        """ Returns a list with all the names of the datasets """
        return [name for name in self.datasets.keys()]

    def keys(self):
        return self.datasets.keys()
