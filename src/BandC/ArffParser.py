from src.BandC.Parser import Parser
import arff
import pandas as pd
from pandas.core.frame import DataFrame


class Arff(Parser):
    """
    An Arff Parser that can automatically detect the correct format.
    """
    def parse_file(self):
        column_names = [attribute[0] for attribute in self.attributes]
        return pd.DataFrame.from_records(self.data, columns=column_names)

    def parse_content(self):
        column_names = [attribute[0] for attribute in self.attributes]
        return pd.DataFrame.from_records(self.data, columns=column_names)

    def detect_dialect(self):
        # The arff package loads an arff file into a dict with the the keys:
        #   description (description of dataset)
        #   relation (name of dataset)
        #   attributes (list of tuples with name and type of attribute)
        #   data (list with the data rows)
        decoder = arff.ArffDecoder()
        if self.contents is None:
            file = open(self.path, 'r')
            weka = decoder.decode(file)
            file.close()
        else:
            weka = decoder.decode(self.decoded_contents)
            # The decoded contents are no longer needed and should not waste memory
            self.decoded_contents = None
        self.name = weka['relation']
        self.description = weka['description']
        # Attribute types are either 'REAL', 'INTEGER', 'NUMERIC' or a list of values (NOMINAL???)
        self.attributes = weka['attributes']
        self.data = weka['data']


if __name__ == '__main__':
    test = Arff('C:/AAA_School/Assignments/BEP/Datasets/Arff Files/glass_nomissing_science.arff',
                verbose=True)
    print(test)
    for key in test.attributes:
        print(key)
    print(test.name)
