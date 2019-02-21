import pandas as pd


class SharedDataFrame:
    """ Shared DataFrame
    Main Abstract Data Type to store, process and use the data
    """
    def __init__(self, df):
        self.df = df
