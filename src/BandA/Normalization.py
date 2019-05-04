from datacleanbot.dataclean import handle_missing
from sklearn import preprocessing
import numpy as np
import pandas as pd


def normalize(df, selection, range = (0,1)):
    """Normalizes the data.

    The user decides which futures are normalized and in what range.

    Parameters
    ----------
    features : list
        List of feature names.

    Xy : array-like
        Numpy array containing the data.

    Returns
    -------

    Xy_normalized : array-like
        Normalized data.

    Xy : array-like
        Original data where no normalization has taken place.
    """

    if np.isnan(df.values).any():
        print("Missing values detected! Please clean missing values first!")
        features, Xy = handle_missing(df.columns, df.values)

    #df = pd.DataFrame(Xy, columns=features)

    #print("What features should be normalized? input = list")
    #eligible_features = df.select_dtypes(include=[np.number]).columns.tolist()
    #print(eligible_features)
    #column_names = input()
    #column_names = list(column_names)
    
    scaler = preprocessing.minmax_scale(Xy, feature_range=range)

    return scaler.fit_transform(Xy)
