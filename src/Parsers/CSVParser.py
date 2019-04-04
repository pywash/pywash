from src.Parsers.Parser import Parser
import pandas as pd
from csv import QUOTE_NONE, QUOTE_MINIMAL

# Import the automatic csv dialect detection package
import os
import sys
sys.path.append(os.path.abspath("C://Users//s163716//Documents//Python Documents//CleverCSV//python"))
import ccsv


class CSV(Parser):
    """
    A CSV Parser that can automatically detect the correct csv format of local csv-files.
    """
    def parse(self):
        # Detect and parse the CSV-file
        if self.verbose:
            print("Returning parsed dataset")
        return self.read_data()

    def detect_parameters(self):
        if self.verbose:
            print("Detecting dialect ...")

        # Create string of dataset
        with open(self.path, 'r') as csv_file:
            reader = csv_file.readlines()
        csv = ''.join(reader)
        # Detect dialect based on string
        dialect = ccsv.Sniffer().sniff(csv, verbose=self.verbose)

        if len(dialect.escapechar) != 1:
            dialect.escapechar = None
        if self.verbose:
            print("Found dialect: " + str(dialect))
        self.__parameters__ = dialect

    def read_data(self):
        if len(self.__parameters__.quotechar) == 0:
            quoting = QUOTE_NONE
        else:
            quoting = QUOTE_MINIMAL
        return pd.read_csv(self.path,
                           sep=self.__parameters__.delimiter,
                           quoting=quoting,
                           quotechar=self.__parameters__.quotechar,
                           escapechar=self.__parameters__.escapechar)


if __name__ == "__main__":
    data = CSV("C://AAA_School/Assignments/BEP/Datasets/Test.csv").parse()
    CSV("C:/AAA_School/Assignments/BEP/Datasets/Test2.csv").parse()
    x = CSV("C:/AAA_School/Assignments/BEP/Datasets/Test3.csv")
    y = x.parse()
    print()
    print(y.head(5))
    print(y.shape)
