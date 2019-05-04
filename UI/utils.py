import pandas as pd
from src.BandA.OutlierDetector import identify_outliers


def handle_outlier_dash(df, selected_algorithms):
    algorithms = ['Isolation Forest', 'Cluster-based Local Outlier Factor', 'Minimum Covariance Determinant (MCD)',
                  'Principal Component Analysis (PCA)', 'Angle-based Outlier Detector (ABOD)',
                  'Histogram-base Outlier Detection (HBOS)', 'K Nearest Neighbors (KNN)', 'Local Outlier Factor (LOF)',
                  'Feature Bagging', 'One-class SVM (OCSVM)']

    if pd.isnull(df).values.any():
        # TODO fix missing data with dc.handle_missing(features, Xy)
        raise ValueError('fix missing data first')

    algorithms = [algorithms[i] for i in selected_algorithms]

    df_sorted, df_styled, df_outliers, df_pred, outliers_count = identify_outliers(df, df.columns, algorithms)
    return df_sorted
