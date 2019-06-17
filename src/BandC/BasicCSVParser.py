from pandas import read_csv
from pandas.core.frame import DataFrame
from src.BandC.Parser import Parser
# TODO, delete this file, deprecated
DeprecationWarning('Deprecated class')


class BasicCSVParser(Parser):
    """ A csv parser without any automation """

    def parse_file(self) -> DataFrame:
        self.detect_dialect()
        return read_csv(self.path, sep=self.__parameters__[0], quotechar=self.__parameters__[1])

    def detect_dialect(self):
        self.__parameters__ = [",", '\"']
