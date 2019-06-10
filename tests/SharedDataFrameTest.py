from unittest import TestCase
from src.PyWash import SharedDataFrame
from src.Exceptions import *
import pandas as pd

verbose = False


class TestDecorators(TestCase):
    """ TestClass for SharedDataFrame methods """

    def test_is_mergeable_column_names(self):
        if verbose:
            print("Testing: is_mergeable_columns")
        df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                            'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
        df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                            'hire_date': [2004, 2008, 2012, 2014]})
        test1 = SharedDataFrame(df=df1, verbose=verbose)
        test2 = SharedDataFrame(df=df2, verbose=verbose)
        self.assertTrue(test1.is_mergeable(test2))

    def test_is_mergeable_common_values(self):
        if verbose:
            print("Testing: is_mergeable_values")
        df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                            'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
        df2 = pd.DataFrame({'names': ['Lisa', 'Bob', 'Jake', 'Sue'],
                            'hire_date': [2004, 2008, 2012, 2014]})
        test1 = SharedDataFrame(df=df1, verbose=verbose)
        test2 = SharedDataFrame(df=df2, verbose=verbose)
        self.assertTrue(test1.is_mergeable(test2))

    def test_is_mergeable_false(self):
        if verbose:
            print("Testing: is_mergeable_false")
        df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                            'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
        df2 = pd.DataFrame({'names': ['Lisa', 'Bob', 'Jake', 'Sue', 'Bobby'],
                            'hire_date': [2004, 2008, 2012, 2014, 2019]})
        test1 = SharedDataFrame(df=df1, verbose=verbose)
        test2 = SharedDataFrame(df=df2, verbose=verbose)
        self.assertFalse(test1.is_mergeable(test2))

    def test_merge_on_column_names(self):
        if verbose:
            print("Testing: merge_on_columns")
        df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                            'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
        df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                            'hire_date': [2004, 2008, 2012, 2014]})
        target = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                               'group': ['Accounting', 'Engineering', 'Engineering', 'HR'],
                               'hire_date': [2008, 2012, 2004, 2014]})
        test1 = SharedDataFrame(df=df1, verbose=verbose)
        test2 = SharedDataFrame(df=df2, verbose=verbose)
        test1.merge_into(test2)
        self.assertTrue(test1.get_dataframe().equals(target), "Successfully merged the 2 DataFrames")

    def test_merge_on_common_values(self):
        if verbose:
            print("Testing: merge_on_values")
        df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                            'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
        df2 = pd.DataFrame({'names': ['Lisa', 'Bob', 'Jake', 'Sue'],
                            'hire_date': [2004, 2008, 2012, 2014]})
        target = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                               'group': ['Accounting', 'Engineering', 'Engineering', 'HR'],
                               'names': ['Bob', 'Jake', 'Lisa', 'Sue'],
                               'hire_date': [2008, 2012, 2004, 2014]})
        test1 = SharedDataFrame(df=df1, verbose=verbose)
        test2 = SharedDataFrame(df=df2, verbose=verbose)
        test1.merge_into(test2)
        if verbose:
            print(test1.get_dataframe())
            print(target)
        self.assertTrue(test1.get_dataframe().equals(target), "Successfully merged the 2 DataFrames")

    def test_merge_on_false(self):
        if verbose:
            print("Testing: merge_false")
        df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                            'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
        df2 = pd.DataFrame({'names': ['Lisa', 'Bob', 'Jake', 'Sue', 'Bobby'],
                            'hire_date': [2004, 2008, 2012, 2014, 2019]})
        test1 = SharedDataFrame(df=df1, verbose=verbose)
        test2 = SharedDataFrame(df=df2, verbose=verbose)
        if verbose:
            print(test1.get_dataframe())
        with self.assertRaises(NotMergableError):
            test1.merge_into(test2)
            if verbose:
                print(test1.get_dataframe())
