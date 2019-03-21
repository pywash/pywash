import pandas as pd


class SharedDataFrame:
    """ Shared DataFrame
    Main Abstract Data Type to store, process and use the data
    """
    def __init__(self, df):
        self.df = df

    def load_data(self):
        pass

    def analyze_data(self):
        """ Determine band value of dataset """
        pass