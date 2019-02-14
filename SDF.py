import pandas as pd


class SDF:
    """ Sharable DataFrame
    Main Abstract Data Type to store, process and use the data
    """
    def __init__(self, dataframe):
        self.df = dataframe
