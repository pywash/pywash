from pandas.core.frame import DataFrame
import base64
import chardet
import os
# TODO incorporate frictionless data


class Parser:
    """
    Base Parser class as parent of all parser classes
    contains __init__ of all parsers to set the filepath and verbose
    contains the property dialect to return the detected dialect of the file
    """
    def __init__(self, file_path: str, contents: str = None, verbose: bool = False):
        """
        Initialize parser by setting the filepath and detecting the dialect of the file

        :param file_path: OS path or URL to parsable file
        :param contents: The dataset as encoded string to be parsed
        :param verbose: Debug boolean
        """
        if verbose:
            print("Initializing parser: " + self.__class__.__name__)
        # Set initial class values
        self.verbose = verbose
        self.path = file_path
        self.name = os.path.splitext(file_path)[0]
        self.contents = contents
        self.encoding = None  # File encoding is currently unknown
        self.attributes = None  # File attributes and types are currently unknown
        self.description = None
        if contents is None:
            # Set the parser to use local file parsing
            self.parse_function = self.parse_file
        else:
            # Set the parser to use string-content based parsing
            self.parse_function = self.parse_content
            # Decode the contents into the dataset string TODO Automatic (effective) decoding
            content_type, content_string = self.contents.split(',')
            decoded = base64.b64decode(content_string)
            self.encoding = chardet.detect(decoded)['encoding']  # Doesn't do anything yet
            self.decoded_contents = decoded.decode('iso-8859-1')  # TODO Use encoding here
            if self.verbose:
                print(self.contents)
                print(self.decoded_contents[:100])
        # Detect the dialect of the dataset based on local implementation
        self.__parameters__ = None  # Initialize dialect
        self.detect_dialect()

    def parse(self) -> DataFrame:
        """ Parses file based on the local implementation of given parse_function """
        if self.verbose:
            print("Parsing dataset: " + str(self.path))
        return self.parse_function()

    def parse_file(self) -> DataFrame:
        """ Parses file based on file-format and file dialect """
        pass

    def parse_content(self) -> DataFrame:
        """ Parses a string of content based on detected dialect """
        pass

    def detect_dialect(self) -> None:
        """
        Detects the dialect (formatting) a file uses to store data, required for parsing correctly
        """
        self.__parameters__ = None

    @property
    def get_dialect(self):
        return self.__parameters__
