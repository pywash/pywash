from src.Parsers.Parser import Parser
import pandas as pd
# Import the automatic csv dialect detection package
import clevercsv as ccsv
from io import StringIO
from csv import QUOTE_NONE, QUOTE_MINIMAL


class CSV(Parser):
    """
    A CSV Parser that can automatically detect the correct csv format of local csv-files.
    """
    def parse_file(self):
        return self.read_data(self.path)

    def parse_content(self):
        return self.read_data(StringIO(self.decoded_contents))

    def string_data(self):
        with open(self.path, 'r') as csv_file:
            self.encoding = csv_file.encoding
            reader = csv_file.readlines()
        return ''.join(reader)

    def detect_dialect(self):
        if self.verbose:
            print("Detecting dialect ...")

        # Create string of dataset
        try:
            csv = self.decoded_contents
        except AttributeError:
            # Load and create a string of the dataset
            csv = self.string_data()

        # Detect dialect based on the decoded dataset string
        dialect = ccsv.Sniffer().sniff(csv, verbose=self.verbose)
        # self.test = dialect.to_csv_dialect() TODO try and use the to_csv_dialect function

        if len(dialect.escapechar) != 1:
            dialect.escapechar = None
        if self.verbose:
            print("Found dialect: " + str(dialect))
        self.__parameters__ = dialect

    def read_data(self, to_parse):
        """
        Uses the detected dialect and a given object to create a DataFrame

        :param to_parse: The object to parse, can be a filepath or BytesIO object
        :return: A DataFrame with the data from the dataset
        """
        if len(self.__parameters__.quotechar) == 0:
            quoting = QUOTE_NONE
        else:
            quoting = QUOTE_MINIMAL
        return pd.read_csv(to_parse,
                           sep=self.__parameters__.delimiter,
                           quoting=quoting,
                           quotechar=self.__parameters__.quotechar,
                           escapechar=self.__parameters__.escapechar)


if __name__ == "__main__":
    data = CSV(file_path="C://AAA_School/Assignments/BEP/Datasets/Test.csv", verbose=True).parse()
    CSV("C:/AAA_School/Assignments/BEP/Datasets/Test2.csv").parse()
    x = CSV("C:/AAA_School/Assignments/BEP/Datasets/Test3.csv")
    y = x.parse_file()
    print()
    print(y.head(5))
    print(y.shape)
