import random

import numpy as np
import pandas as pd
import seaborn as sns
from pyod.models.knn import KNN as knn
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.lscp import LSCP


def estimate_contamination(df):
    df_numeric = df.select_dtypes(include=[np.number])
    total_length = len(df_numeric)  # total length of the dataframe, used for computing contamination later
    dict_outliers = {}
    df_union = pd.DataFrame()
    # to estimate the propotion of outliers
    for i, col in enumerate(df_numeric.columns):
        # first detect outliers in each column
        # keep only the ones that are out of +3 to -3 standard deviations in the column 'Data'.
        dict_outliers[col] = df_numeric[~(np.abs(df_numeric[col] - df_numeric[col].mean()) < (
                3 * df_numeric[col].std()))]  # ~ means the other way around
        # combine all the rows containing outliers in one feature
        df_union = df_union.combine_first(dict_outliers[col])

    max_outliers = len(df_union)
    contamination = max_outliers / total_length

    # handle edge cases
    if contamination == 0:
        contamination = 0.001
    elif contamination > 0.5:
        contamination = 0.5
    return contamination


def identify_outliers(df, features, contamination=0.1, algorithms=['Isolation Forest']):
    """Cleans the outliers.

    Outlier detection using LSCP: Locally selective combination in parallel outlier ensembles.
    https://arxiv.org/abs/1812.01528


    Parameters
    ----------
    features : list
        List of feature names.

    df : DataFrame
        The data to be examined.

    contamination : float in (0., 0.5)
        the proportion of outliers in the data set.

    algorithms: list
        list with at the names of least 2 algorithms to be used during LSCP. A list of supported algorithms:

        ['Isolation Forest', 'Cluster-based Local Outlier Factor', 'Minimum Covariance Determinant (MCD)',
                  'Principal Component Analysis (PCA)', 'Angle-based Outlier Detector (ABOD)',
                  'Histogram-base Outlier Detection (HBOS)', 'K Nearest Neighbors (KNN)', 'Local Outlier Factor (LOF)',
                  'Feature Bagging', 'One-class SVM (OCSVM)']



    Returns
    -------

    df_sorted : DataFrame
        Original data with 3 new columns: anomaly_score, probability and prediction. Sorted on descending anomaly score.

    df_styled: DataFrame
        Styled version of df_sorted for use in Jupyter Notebook (i.e. display(df_styled)).
    """

    df_numeric = df.select_dtypes(include=[np.number])  # keep only numeric type features
    X = np.asarray(df_numeric)

    classifiers = {'Isolation Forest': IForest,
                   'Cluster-based Local Outlier Factor': CBLOF,
                   'Minimum Covariance Determinant (MCD)': MCD,
                   'Principal Component Analysis (PCA)': PCA,
                   'Angle-based Outlier Detector (ABOD)': ABOD,
                   'Histogram-base Outlier Detection (HBOS)': HBOS,
                   'K Nearest Neighbors (KNN)': knn,
                   'Local Outlier Factor (LOF)': LOF,
                   'Feature Bagging': FeatureBagging,
                   'One-class SVM (OCSVM)': OCSVM,
                   }

    if len(algorithms) > 1:
        selected_classifiers = [classifiers[x]() for x in algorithms]
        clf = LSCP(selected_classifiers, contamination=contamination)
    else:
        clf = classifiers[algorithms[0]](contamination=contamination)

    clf.fit(X)
    y_pred = clf.predict(X)

    y_predict_proba = clf.predict_proba(X, method='unify')
    y_predict_proba = [item[1] for item in y_predict_proba]

    outlier_index, = np.where(y_pred == 1)

    anomaly_score = clf.decision_function(X)
    anomaly_score = pd.DataFrame(anomaly_score, columns=['anomaly_score'])

    y_predict_proba = pd.DataFrame(y_predict_proba, columns=['probability'])
    prediction = pd.DataFrame(y_pred, columns=['prediction'])

    df.columns = features
    df_with_anomaly_score = pd.concat([df, anomaly_score, y_predict_proba, prediction], axis=1)

    df_sorted = df_with_anomaly_score.sort_values(by='anomaly_score', ascending=False)
    cm = sns.diverging_palette(220, 10, sep=80, n=7, as_cmap=True)
    df_styled = df_sorted.style.background_gradient(cmap=cm, subset=['anomaly_score']) \
        .apply(lambda x: ['background: MistyRose' if x.name in outlier_index.tolist() else '' for i in x], axis=1,
               subset=df_sorted.columns[:-3])

    return df_sorted, df_styled
