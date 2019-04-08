from src import Parsers as Parser


class SharedDataFrame:
    """ Shared DataFrame
    Main Abstract Data Type to store, process and use the data
    """
    def __init__(self, file_path: str, verbose:bool = False):
        self.file_path = file_path
        self.parser = Parser.assign_parser(file_path, verbose)

    def load_data(self):
        self.data = self.parser.parse()

    def analyze_data(self):
        """ Determine band value of dataset """
        pass