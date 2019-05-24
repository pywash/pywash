import datacleanbot.dataclean as dc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from IPython.core.display import display, HTML
import time
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


def handle_outlier(features, Xy):
    """Cleans the outliers.

    Recommends the algorithm to the user to detect the outliers and
    presents the outliers to the user in effective visualizations.
    The user can decides whether or not to keep the outliers.

    Parameters
    ----------
    features : list
        List of feature names.

    Xy : array-like
        Numpy array. Both training vectors and target are required.

    Returns
    -------

    Xy_no_outliers : array-like
        Cleaned data where outliers are dropped.

    Xy : array-like
        Original data where outliers are not found or kept.
    """
    algorithms = ['Isolation Forest', 'Cluster-based Local Outlier Factor', 'Minimum Covariance Determinant (MCD)',
                  'Principal Component Analysis (PCA)', 'Angle-based Outlier Detector (ABOD)',
                  'Histogram-base Outlier Detection (HBOS)', 'K Nearest Neighbors (KNN)', 'Local Outlier Factor (LOF)',
                  'Feature Bagging', 'One-class SVM (OCSVM)']

    display(HTML('<h2>Outliers</h2>'))

    if np.isnan(Xy).any():
        print("Missing values detected! Please clean missing values first!")
        features, Xy = dc.handle_missing(features, Xy)

    print("")
    print("What setting?")
    print("a.Regular b.Fast c.Full d.custom")
    ans = input()
    if ans == 'a':
        algorithms = algorithms[0:8]
        print("Using:")
        print(algorithms)
    if ans == 'b':
        algorithms = algorithms[0:4]
        print("Using:")
        print(algorithms)
    if ans == 'c':
        print("Using:")
        print(algorithms)
    if ans == 'd':
        print("Please select algorithms from the following list")
        print("")
        for i in range(0, 10):
            print(i, algorithms[i])
        print("")
        print("Enter a list separated by spaces (i.e. '0 3 5')")
        s = input()
        numbers = list(map(int, s.split()))
        algorithms = [algorithms[i] for i in numbers]

    df = pd.DataFrame(Xy)
    display(HTML('<h4>Visualize Outliers ... </h4>'))
    df_sorted, df_styled, df_outliers, df_pred, outliers_count = identify_outliers(df, features, algorithms)
    dc.visualize_outliers_scatter(df, df_pred)
    print("")
    display(HTML('<h4>Drop Outliers ... </h4>'))
    time.sleep(0.05)
    print("Do you want to drop outliers?")
    print("")
    print("a. Yes, drop all red highlighted outliers")
    print("b. Yes, drop all outliers with probability 1")
    print("c. Yes, custom")
    print("d. No")
    print("")
    ans = input()
    if ans == 'a':
        print("Outliers are dropped.")
        df_no_outliers = dc.drop_outliers(df, df_outliers)
        Xy_no_outliers = df_no_outliers.values
        return Xy_no_outliers
    if ans == 'b':
        df_outliers = df_outliers[df_outliers['probability'] >= 0.9999995]
        df_no_outliers = dc.drop_outliers(df, df_outliers)
        Xy_no_outliers = df_no_outliers.values
        print("Deleted the following rows")
        display(df_outliers)
        return Xy_no_outliers
    if ans == 'c':
        ans = input("Enter list with indexes to be removed: [index #1,index #2 etc]")
        df_no_outliers = dc.drop_outliers(df, df[-ans])
        Xy_no_outliers = df_no_outliers.values
        return Xy_no_outliers
    else:
        print("Outliers are kept.")
        return Xy


def identify_outliers(df, features, algorithms):
    """Identifies outliers in multi dimension.

    Dataset has to be parsed as numeric beforehand.
    """

    df_exclude_target = df.iloc[:, :-1]
    df_numeric = df_exclude_target.select_dtypes(include=[np.number])  # keep only numeric type features
    total_length = len(df_numeric)  # total length of the dataframe, used for computing contamination later

    outliers_count = np.zeros(len(df_numeric.columns))  # number of outliers of each feature
    dict_outliers = {}
    df_union = pd.DataFrame()
    for i, col in enumerate(df_numeric.columns):
        #         if(df_numeric[col].dtype in [np.number]): # bug! to be figured out

        # first detect outliers in each column
        # keep only the ones that are out of +3 to -3 standard deviations in the column 'Data'.
        dict_outliers[col] = df_numeric[~(np.abs(df_numeric[col] - df_numeric[col].mean()) < (
                3 * df_numeric[col].std()))]  # ~ means the other way around
        # combine all the rows containing outliers in one feature
        df_union = df_union.combine_first(dict_outliers[col])
    #             print(dict_outliers[col])

    # Two options to estimate the propotion of outliers
    # One is to take the number of outliers in the feature containing most outliers
    # The other is to take the length of the union of rows containing outliers in any feature
    #     print(outliers_count)
    #     print(df_union)
    #     max_outliers = max(outliers_count)
    max_outliers = len(df_union)

    contamination = max_outliers / total_length

    if contamination == 0:
        contamination = 0.001
    elif contamination > 0.5:
        contamination = 0.5

    X = np.asarray(df_numeric)

    classifiers = {'Isolation Forest': IForest(),
                   'Cluster-based Local Outlier Factor': CBLOF(),
                   'Minimum Covariance Determinant (MCD)': MCD(),
                   'Principal Component Analysis (PCA)': PCA(),
                   'Angle-based Outlier Detector (ABOD)': ABOD(),
                   'Histogram-base Outlier Detection (HBOS)': HBOS(),
                   'K Nearest Neighbors (KNN)': knn(),
                   'Local Outlier Factor (LOF)': LOF(),
                   'Feature Bagging': FeatureBagging(),
                   'One-class SVM (OCSVM)': OCSVM(),
                   }

    selected_classifiers = [classifiers[x] for x in algorithms]

    clf = LSCP(selected_classifiers, contamination=contamination)
    clf.fit(X)
    y_pred = clf.predict(X)

    y_predict_proba = clf.predict_proba(X, method='unify')
    y_predict_proba = [item[1] for item in y_predict_proba]

    outlier_index, = np.where(y_pred == 1)
    df_outliers = df_numeric.loc[outlier_index.tolist()]
    anomaly_score = clf.decision_function(X)

    anomaly_score = pd.DataFrame(anomaly_score, columns=['anomaly_score'])
    y_predict_proba = pd.DataFrame(y_predict_proba, columns=['probability'])
    prediction = pd.DataFrame(y_pred, columns=['prediction'])

    df.columns = features
    df_with_anomaly_score = pd.concat([df, anomaly_score, y_predict_proba, prediction], axis=1)
    df_outliers = pd.concat([df_outliers, y_predict_proba], axis=1)

    df_sorted = df_with_anomaly_score.sort_values(by='anomaly_score', ascending=False)
    cm = sns.diverging_palette(220, 10, sep=80, n=7, as_cmap=True)
    # TODO Change style for big datasets (do not display everything)
    df_styled = df_sorted.style.background_gradient(cmap=cm, subset=['anomaly_score']) \
        .apply(lambda x: ['background: MistyRose' if x.name in outlier_index.tolist() else '' for i in x], axis=1,
               subset=df_sorted.columns[:-3]) \
        .apply(dc.highlight_outlier, subset=df_sorted.columns[:-3])

    display(df_styled)
    return df_sorted, df_styled, df_outliers, prediction, outliers_count
