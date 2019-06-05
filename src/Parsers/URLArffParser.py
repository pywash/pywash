from src.Parsers.Parser import Parser
import requests
import urllib.request as req
from csv import QUOTE_NONE, QUOTE_MINIMAL
import pandas as pd
import arff
import io
#from scipy.io import arff
import time



class URLARFF(Parser):
    """
    A CSV Parser that can automatically detect the correct csv format of online datasets.
    """
    def parse_file(self):
        return self.read_data()

    def detect_dialect(self):
        self.verbose = True
        self.name = self.name.split('/')[-1]

        content = self.decode_page()
        #print(content)
        #arff.load(content)
        """
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
            print("No result from CleverCSV")"""

    def decode_page(self):
        """ Get the contents of a page, assuming UTF-8 """
        # TODO Create encoding detection for online files
        results = req.urlopen(self.path)
        print(results.read().decode('utf-8'))
        print(str(results.read().decode('utf-8')))
        time.sleep(0.2)
        #print(arff.loads(str(results.read().decode('utf-8'))))
        #print(arff.load(io.StringIO(results.read().decode('utf-8'))))
        page = requests.get(self.path)
        print(page.content.replace('\r\n', '\n'))
        print(arff.load(page.content.replace('\r\n', '\n')))
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

    def export(self, df, file_path: str):
        df.to_csv(file_path)


if __name__ == '__main__':
    # Test files
    # https://raw.githubusercontent.com/renatopp/arff-datasets/master/boolean/xor.arff
    # https://github.com/renatopp/arff-datasets/blob/master/agridata/eucalyptus.arff
    xxx = URLARFF("https://raw.githubusercontent.com/renatopp/arff-datasets/master/boolean/xor.arff")
    #arf = URLARFF("https://github.com/renatopp/arff-datasets/blob/master/agridata/eucalyptus.arff",
    #              verbose=True)
