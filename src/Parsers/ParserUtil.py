from src.Parsers import *
from src.Exceptions import FileFormatNotFound

__parsers__ = {'.csv': CSV}
__url_parsers__ = {'.csv': URLCSV}


def assign_parser(file_path: str, contents: str = None, verbose: bool = False) -> callable:
    """ Allocate a specific parser to a file_path

    :param file_path: The file path of the dataset to parse
    :param contents: The dataset as a string
    :param verbose: True for output
    :return: A parser object which is able to parse the dataset
    """
    # Check file path to see if we need a local or an url parser
    parsers = __parsers__
    if file_path.startswith('https:'):
        parsers = __url_parsers__

    # Find the correct parser for the file
    for parser in parsers:
        # Check if we have implemented a parser for this file
        if file_path.endswith(parser):
            # Check if the dataset has been given as a string
            if contents is None:
                return __parsers__[parser](file_path=file_path,
                                           verbose=verbose)
            else:
                return __parsers__[parser](file_path=file_path,
                                           contents=contents,
                                           verbose=verbose)

    # When the file format is not in our list of parable formats
    raise FileFormatNotFound("File format of file: " + file_path + " is unknown")


if __name__ == "__main__":
    p = assign_parser("C:/AAA_School/Assignments/BEP/Datasets/Test.csv", True)
    print()
    print(p)
    print(p.get_dialect())
    print()
    print(p.parse_file().head(5))
    print('Done')
