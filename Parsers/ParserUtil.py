from Parsers.BasicCSVParser import BasicCSVParser

parsers = {'.csv': BasicCSVParser}


def assign_parser(file_path: str) -> callable:
    """ Allocate a specific parser to a file_path

    :param file_path: The file path of the dataset to parse
    :return: A parser object which is able to parse the dataset
    """
    for parser in parsers:
        if parser in file_path:
            return parsers[parser](file_path)


if __name__ == "__main__":
    print("Parsing test dataset:")
    p = assign_parser("C:\AAA School\Assignments\BEP\Datasets\Test.csv").parse()
    print(p.head(3))
    print('Done')
