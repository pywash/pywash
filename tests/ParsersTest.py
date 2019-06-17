from unittest import TestCase
from src.PyWash import SharedDataFrame
from src.BandC import *
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

    def test_arff_export(self):
        if verbose:
            print("Testing: test_arff_export")
        sdf = SharedDataFrame('C:/AAA_School/Assignments/BEP/Datasets/Arff Files/arrhythmia_missing.arff')
        sdf.export('export_arff.arff')
