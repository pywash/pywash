from pandas.core.frame import DataFrame
# TODO incorporate frictionless data


class Parser:
    """
    Base Parser class as parent of all parser classes
    contains __init__ of all parsers to set the filepath and verbose
    contains the property dialect to return the detected dialect of the file
    """
    def __init__(self, file_path: str, verbose: bool = False) -> None:
        """
        Initialize parser by setting the filepath and detecting the dialect of the file

        :param file_path: OS path or URL to parsable file
        :param verbose:
        """
        if verbose:
            print("Initializing parser: " + self.__class__.__name__)
        self.verbose = verbose
        self.path = file_path
        self.__parameters__ = None  # Initialize dialect as None
        self.detect_parameters()  # Detect the dialect based on local implementation

    def parse(self) -> DataFrame:
        """ Parses file based on file-format and file dialect """
        pass

    def detect_parameters(self) -> None:
        """
        Detects the dialect (formatting) a file uses to store data, required for parsing correctly
        """
        self.__parameters__ = None

    @property
    def dialect(self):
        return self.__parameters__
