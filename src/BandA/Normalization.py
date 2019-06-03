import os
os.environ['R_USER'] = 's163716'
from datacleanbot.dataclean import handle_missing
from sklearn import preprocessing
import pandas as pd


def normalize(df, selection, range=(0, 1)):
    """Normalizes the data.

    The user decides which futures are normalized and in what range.

    Parameters
    ----------

    df : DataFrame
        DataFrame containing the data.

    selection : list
        List of columns to be normalized.

    feature_range : default=(0, 1)
        Desired range of transformed data.

    Returns
    -------

    df : array-like
        Original data where desired columns have been normalized.
    """

    if pd.isnull(df).values.any():
        # TODO implement handle_missing for dash
        print("Missing values detected! Please clean missing values first!")
        features, Xy = handle_missing(df.columns, df.values)

    df[selection] = pd.DataFrame(preprocessing.minmax_scale(df[selection], feature_range=range))

    return df
