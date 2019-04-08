from pandas import read_csv
from pandas.core.frame import DataFrame
from src.Parsers.Parser import Parser


class BasicCSVParser(Parser):
    """ A csv parser without any automation """

    def parse(self) -> DataFrame:
        self.detect_parameters()
        return read_csv(self.path, sep=self.__parameters__[0], quotechar=self.__parameters__[1])

    def detect_parameters(self):
        self.__parameters__ = [",", '\"']
