from src.Parsers.Parser import Parser
import urllib.request as req
import pandas as pd
import arff
import io


class URLARFF(Parser):
    """
    An ARFF Parser that can automatically loads an online dataset and can load it into a dataframe.
    """
    def parse_file(self):
        column_names = [attribute[0] for attribute in self.attributes]
        return pd.DataFrame.from_records(self.data, columns=column_names)

    def detect_dialect(self):
        if self.verbose:
            print('Loading arff file from internet')
        try:
            arff_dict = self.decode_page()
        except:
            print('Arff file request failed')

        self.name = arff_dict['relation']
        self.description = arff_dict['description']
        self.attributes = arff_dict['attributes']
        self.data = arff_dict['data']

    def decode_page(self):
        """ Get the contents of a page, assuming UTF-8 """
        # TODO Create encoding detection for online files
        results = req.urlopen(self.path)
        return arff.load(io.StringIO(results.read().decode('utf-8')))

    def export(self, df, file_path: str):
        df.to_csv(file_path)


if __name__ == '__main__':
    # Test files
    # https://raw.githubusercontent.com/renatopp/arff-datasets/master/boolean/xor.arff
    # https://raw.githubusercontent.com/renatopp/arff-datasets/master/agridata/eucalyptus.arff
    xxx = URLARFF("https://raw.githubusercontent.com/renatopp/arff-datasets/master/boolean/xor.arff")
    arf = URLARFF("https://raw.githubusercontent.com/renatopp/arff-datasets/master/agridata/eucalyptus.arff",
                  verbose=True)
