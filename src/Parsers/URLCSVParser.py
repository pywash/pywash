import pandas as pd
import requests
from csv import QUOTE_NONE, QUOTE_MINIMAL
from src.Parsers.Parser import Parser
# Import the automatic csv dialect detection package
from src.Parsers import ccsv


class URLCSV(Parser):
    """
    A CSV Parser that can automatically detect the correct csv format of online datasets.
    """
    def parse(self):
        if self.verbose:
            print("Returning parsed dataset")
        return self.read_data()

    def detect_parameters(self):
        if self.verbose:
            print("Detection dialect ...")
        content = self.decode_page()
        try:
            dialect = ccsv.Sniffer().sniff(content, verbose=self.verbose)
            if self.verbose:
                print("Found dialect: " + str(dialect))
            if len(dialect.escapechar) != 1:
                dialect.escapechar = None
            self.__parameters__ = dialect
            if self.verbose:
                print("Found dialect: " + str(dialect))

        except ccsv.Error:
            print("No result from CleverCSV")

    def decode_page(self):
        """ Get the contents of a page, assuming UTF-8 """
        # TODO Create encoding detection for online files
        page = requests.get(self.path)
        return page.content.decode('utf-8')

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
    test_1 = "https://raw.githubusercontent.com/agh-glk/pyconpl2013-nlp/37f6f50a45fc31c1a5ad25010fff681a8ce645b8/gsm.csv"
    test_2 = "https://raw.githubusercontent.com/queq/just-stuff/c1b8714664cc674e1fc685bd957eac548d636a43/pov/TopFixed/build/project_r_pad.csv"
    # URLCSV appears not to work on test 2.
    p = URLCSV(test_1, True)
    x = p.parse()
    print()
    print(x.head(5))
    print(x.shape)
