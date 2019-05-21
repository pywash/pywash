from src.Parsers.ParserUtil import assign_parser
from src.Exceptions import *
from pandas.core.frame import DataFrame


class SharedDataFrame:
    """ Shared DataFrame
    Main Abstract Data Type to store, process and use the data
    """
    def __init__(self, file_path: str = None, contents: str = None, df: DataFrame = None,
                 verbose: bool = False):
        """ Initializes the SharedDataFrame
        Can be given a path to a file to parse
         or a dataset as string needed to be parsed
         or a parsed DataFrame can be given to be used
        """
        self.verbose = verbose
        self.file_path = file_path
        self.data = None
        self.parser = None
        self.score = None
        # When a path to a file or the contents are given, parse the file and load the data
        if file_path is not None:
            self.parser = assign_parser(file_path=file_path, contents=contents, verbose=verbose)
            self._load_data()
        # When a DataFrame is given, set the DataFrame as the SharedDataFrame data
        elif df is not None:
            self.set_data(df)
        self.name = self.parser.name

    def __repr__(self):
        # TODO, create representation
        NotImplementedError("Create")

    def __str__(self) -> str:
        # SharedDataFrames are represented by their file_name and the dimensions of the data
        return str(self.file_path) + " " + str(self.data.shape)

    def _load_data(self):
        self.data = self.parser.parse()

    def set_data(self, df):
        """ Sets an pre-parsed DataFrame as the data of the SharedDataFrame """
        self.data = df

    def get_dataframe(self):
        return self.data

    def analyze_data(self):
        """ Determine band value of dataset """
        pass

    def get_datascore(self):
        """ Return the band value of the dataset """
        return self.score

    # Merge functions #####
    def is_mergeable(self, other_sdf) -> bool:
        """ Checks if 2 SharedDataFrames are mergeable.
        DataFrames are mergeable when they either have one or more columns names in common
         or when a combination of unique values of one or more columns exists in both DataFrames.
        """
        # Common columns names
        if self._has_common_column_names(other_sdf):
            if self.verbose:
                print("{} and {} have common columns names".format(self, other_sdf))
            return True
        # Find columns with the same unique values
        if len(self._find_common_column_values(other_sdf)[0]) > 0:
            return True
        return False

    def _has_common_column_names(self, other_sdf) -> bool:
        """ Checks all column names with all column names from another SDF on duplicates """
        for column in self.data:
            if column in other_sdf.get_dataframe():
                return True
        return False

    def _find_common_column_values(self, other_sdf) -> tuple:
        """
        Finds all columns from this and another SharedDataFrame that contain the exact same values
        """
        common_columns = []
        other_common_columns = []
        # Investigate 1 column from each dataset at a time
        for column in self.data:
            for other_column in other_sdf.get_dataframe():
                # Check if the unique values of both investigated columns are equal
                if set(self.data[column].unique()) == set(other_sdf.get_dataframe()[other_column].unique()):
                    common_columns.append(column)
                    other_common_columns.append(other_column)
        return common_columns, other_common_columns

    def merge(self, other_sdf):
        """ Merges, if possible, another SharedDataFrame with this DataFrame """
        if not self.is_mergeable(other_sdf):
            raise NotMergableError(
                "Attempting to merge non-mergeable DataFrames {} and {}".format(self, other_sdf))
        elif self._has_common_column_names(other_sdf):
            return self.data.merge(other_sdf.get_dataframe())
        else:
            left_columns, right_columns = self._find_common_column_values(other_sdf)
            return self.data.merge(other_sdf.get_dataframe(),
                                   left_on=left_columns, right_on=right_columns)

    def merge_into(self, other_sdf):
        """ Merges another SharedDataFrame into the current DataFrame """
        self.data = self.merge(other_sdf)
