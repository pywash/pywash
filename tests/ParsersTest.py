from unittest import TestCase
from src.SharedDataFrame import SharedDataFrame
from src.Parsers import *
from pathlib import Path
from pandas.core.frame import DataFrame

verbose = False


class TestDecorators(TestCase):
    """ TestClass for SharedDataFrame methods """

    # Passed 1103 of 1136 tests (33 failures)
    def test_arff_parser(self):
        if verbose:
            print("Testing: test_arff_parser")
        for path in Path("C:/AAA_School/Assignments/BEP/Datasets/Arff Files").glob('**/*.arff'):
            with self.subTest(path=path):
                try:
                    sdf = SharedDataFrame(str(path))
                    self.assertIsInstance(sdf.parser, Arff, "Used the Arff parser")
                    self.assertIsInstance(sdf.get_dataframe(), DataFrame, "Created pandas dataframe")
                except Exception as e:
                    print([attribute[1] for attribute in sdf.parser.attributes])
                    self.assertTrue(False, 'Arff file ({}) loading error: {}'.format(sdf.name, e))