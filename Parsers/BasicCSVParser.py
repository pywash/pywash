import pandas as pd
from Parsers.Parser import Parser


class BasicCSVParser(Parser):
    """ A csv parser without any automation """

    def parse(self) -> pd.DataFrame:
        options = self.parse_parameters()
        return pd.read_csv(self.path, sep=options[0], quotechar=options[1])

    def parse_parameters(self):
        return [",", '\"']
