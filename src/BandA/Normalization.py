from sklearn import preprocessing
import pandas as pd


def normalize(df, selection, setting, range=(0, 1)):
    """Normalizes the data.

    The user decides which futures are scaled and in what range.

    Parameters
    ----------

    df : DataFrame
        DataFrame containing the data.

    selection : list
        List of columns to be normalized.

    setting : string
        'normalize' or 'standardize'

    feature_range : default=(0, 1)
        Desired range of transformed data.

    Returns
    -------

    df : array-like
        Original data where desired columns have been scaled.
    """

    if pd.isnull(df).values.any():
        # TODO implement handle_missing for dash
        print("Missing values detected! Please clean missing values first!")

    if setting == 'normalize':
        df[selection] = pd.DataFrame(preprocessing.minmax_scale(df[selection], feature_range=range))
    if setting == 'standardize':
        scaler = preprocessing.StandardScaler()
        df[selection] = pd.DataFrame(scaler.fit_transform(df[selection]))

    return df
