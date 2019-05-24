import numpy as np
import pandas as pd
from fancyimpute import KNN, MatrixFactorization, IterativeImputer
from sklearn.preprocessing import Imputer
import datacleanbot.dataclean as dc
import os
import sys


# hide print
class NoStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush();
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush();
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def identify_missing(df=None, na_values=['n/a', 'na', '--', '?']):
    """Detect missing values.
    Identify the common missing characters such as 'n/a', 'na', '--'
    and '?' as missing. User can also customize the characters to be
    identified as missing.
    Parameters
    ----------
    df : DataFrame
        Raw data formatted in DataFrame.
    Returns
    -------
    flag : bool
        Indicates whether missing values are detected.
        If true, missing values are detected. Otherwise not.
    """
    for value in na_values:
        df = df.replace(value, np.nan)

    # flag indicates whether any missing value is detected
    flag = df.isnull().values.any()
    return flag


def identify_missing_mechanism(df=None):
    """Tries to guess the missing mechanism of the dataset.
    Missing mechanism is not really testable. There may be reasons to
    suspect that the dataset belongs to one missing mechanism based on
    the missing correlation between features, but the result is not
    definite. Relevant information are provided to help the user make
    the decision.
    Three missng mechanisms to be guessed:
    MCAR: Missing completely at ramdom
    MAR: Missing at random
    MNAR: Missing not at random (not available here, normally involes field expert)
    Parameters
    ----------
    df : DataFrame
        Raw data formatted in DataFrame.
    """
    # Pearson correlation coefficient between every 2 features
    #     print("")
    #     print("Missing correlation (Pearson correlation coefficient) between every 2 features")
    #     display(df.isnull().corr())
    df2 = df.iloc[:, :-1].copy()
    missing_columns = df2.columns[df2.isnull().any(axis=0)]  # columns containing missing values
    # replace nan as true, otherwise false for features containing missing values
    df2[df2.columns[df2.isnull().any(axis=0)]] = df2[df2.columns[df2.isnull().any(axis=0)]].isnull()
    df2[missing_columns] = df2[missing_columns].astype(int)  # replace true as 1, false as 0
    df_missing_corr = df2.corr()[
        missing_columns]  # compute correlations between features containing missing values and other features
    print("Missing correlation between features containing missing values and other features")
    # display(df_missing_corr)
    flag_mar = False
    # test if there is some correlation of a value being missed in feature and the value of any other of the features
    for col in df_missing_corr:
        list_high_corr = []
        list_high_corr = list_high_corr + (df_missing_corr[col].index[df_missing_corr[col] > 0.6].tolist())
        list_high_corr.remove(int(col))
        #         print(list_high_corr)
        if list_high_corr:
            flag_mar = True
    if flag_mar:
        print('Missing mechanism is probably missing at random')
    else:
        print('Missing mechanism is probably missing completely at random')


#     tri_lower_no_diag = np.tril(df.isnull().corr(), k=-1)
#     # if any 2 features highly missing correlated
#     if (tri_lower_no_diag > 0.6).any() or (tri_lower_no_diag < -0.6).any():
#         display(HTML('<bold>Missing mechanism is highly possible to be missing at random</bold>'))
#     elif (tri_lower_no_diag > -0.2).all() and (tri_lower_no_diag < 0.2).all():
#         display(HTML('<bold>Missing mechanism is highly possible to be missing completely at random</bold>'))
#     else:
#         display(HTML('<bold>Missing mechanism is hard to guess</bold>'))



def missing_preprocess(features, df=None):
    """Drops the redundant information.
    Redundant information is dropped before imputation. Detects and
    drops empty rows. Detects features and instances with extreme large
    proportion of missing data and reports to the user.
    Parameters
    ----------
    features : list
        List of feature names.
    df : DataFrame
    Returns
    -------
    df : DataFrame
        New DataFrame where redundant information may have been deleted.
    features_new: list
        List of feature names after preprocessing.
    """

    # number of missing in each row
    #     print(df.isnull().sum(axis=1))

    # number of missing in each feature
    #     print(df.isnull().sum())

    # number of instances
    num_instances = df.shape[0]
    # number of features
    num_features = df.shape[1]

    # detect empty rows
    if any(df.isnull().sum(axis=1) == num_features):
        print(df[df.isnull().sum(axis=1) == num_features])
        print("Above empty rows are detected and removed \n")
        df = df.dropna(how='all')  # remove empty rows

    large_missing_cols = []  # list of columns with extreme large proportion of missing data
    for col in df.columns[:-1]:  # exclude target class
        if df[col].isnull().sum() > 0.9 * num_instances:
            large_missing_cols.append(col)
    if large_missing_cols:
        print("Feature {} has extreme large proportion of missing data".format(large_missing_cols))
        ans = input('Do you want to delete the above features? [y/n]')
        if ans == 'y':
            df.drop(large_missing_cols, 1, inplace=True)
        else:
            pass
    print(df.columns)
    features_new = df.columns.values
    return df, features_new


'''
def compute_imputation_score(Xy):
    """Computes score of the imputation by applying simple classifiers.
    The following simple learners are evaluated:
    Naive Bayes Learner;
    Linear Discriminant Learner;
    One Nearest Neighbor Learner;
    Decision Node Learner.
    Parameters
    ----------
    Xy : array-like
        Complete numpy array of the dataset. The training array X has to be imputed
        already, and the target y is required here and not optional in order to
        predict the performance of the imputation method.
    Returns
    -------
    imputation_score : float
        Predicted score of the imputation method.
    """
    X = Xy[:, :-1]
    #     print(X.dtype)
    y = Xy[:, -1]
    y = y.astype('int')
    #     print(y.dtype)
    scores = []
    naive_bayes = GaussianNB()
    decision_node = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=1, random_state=0)
    linear_discriminant_analysis = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    one_nearest_neighbor = KNeighborsClassifier(n_neighbors=1)
    classifiers = [naive_bayes, decision_node, linear_discriminant_analysis, one_nearest_neighbor]
    for classifier in classifiers:
        # compute accuracy score for each simple classifier
        score = np.mean(cross_val_score(classifier, X, y, cv=5, scoring='accuracy', n_jobs=-1))
        #         print("Score of {} is {}".format(classifier, score))
        scores.append(score)
    imputation_score = np.mean(scores)
    return imputation_score
'''


def deal_mcar(df):
    """Deal with missing data with missing completely at random pattern."""
    # number of instances
    num_instances = df.shape[0]

    # number of rows containing missing
    num_missing_instances = df.isnull().sum(axis=1).astype(bool).sum()

    # missing percentage
    missing_percentage = num_missing_instances / num_instances
    print("Missing percentage is {}".format(missing_percentage))

    if missing_percentage < 0.05:
        recommend = 'list deletion'
    else:
        Xy_incomplete = df.values
        # mean
        Xy_filled_mean = Imputer(missing_values=np.nan, strategy='mean').fit_transform(Xy_incomplete)
        score_mean = dc.compute_imputation_score(Xy_filled_mean)
        print("Imputation score of mean is {}".format(score_mean))
        # mode
        Xy_filled_mode = Imputer(missing_values=np.nan, strategy='most_frequent').fit_transform(Xy_incomplete)
        score_mode = dc.compute_imputation_score(Xy_filled_mode)
        print("Imputation score of mode is {}".format(score_mode))
        # knn
        with NoStdStreams():
            Xy_filled_knn = KNN().fit_transform(Xy_incomplete);
        score_knn = dc.compute_imputation_score(Xy_filled_knn)
        print("Imputation score of knn is {}".format(score_knn))
        # matrix factorization
        with NoStdStreams():
            Xy_filled_mf = MatrixFactorization().fit_transform(Xy_incomplete);
        score_mf = dc.compute_imputation_score(Xy_filled_mf)
        print("Imputation score of matrix factorization is {}".format(score_knn))
        # multiple imputation
        with NoStdStreams():
            Xy_filled_ii = IterativeImputer().fit_transform(Xy_incomplete)
        score_ii = dc.compute_imputation_score(Xy_filled_ii)
        print("Imputation score of multiple imputation is {}".format(score_ii))

        score_dict = {'mean': score_mean, 'mode': score_mode, 'knn': score_knn,
                      'matrix factorization': score_mf, 'multiple imputation': score_ii}
        print("Imputation method with the highest socre is {}".format(max(score_dict, key=score_dict.get)))
        recommend = max(score_dict, key=score_dict.get)
    return recommend


def deal_mar(df):
    """Deal with missing data with missing at random pattern."""

    Xy_incomplete = df.values

    # knn
    with NoStdStreams():
        Xy_filled_knn = KNN().fit_transform(Xy_incomplete);
    score_knn = dc.compute_imputation_score(Xy_filled_knn)
    print("Imputation score of knn is {}".format(score_knn))
    # matrix factorization
    with NoStdStreams():
        Xy_filled_mf = MatrixFactorization().fit_transform(Xy_incomplete);
    score_mf = dc.compute_imputation_score(Xy_filled_mf)
    print("Imputation score of matrix factorization is {}".format(score_knn))
    # multiple imputation
    with NoStdStreams():
        Xy_filled_ii = IterativeImputer().fit_transform(Xy_incomplete)
    score_ii = dc.compute_imputation_score(Xy_filled_ii)
    print("Imputation score of multiple imputation is {}".format(score_ii))

    score_dict = {'knn': score_knn,
                  'matrix factorization': score_mf, 'multiple imputation': score_ii}
    print("Imputation method with the highest socre is {}".format(max(score_dict, key=score_dict.get)))
    recommend = max(score_dict, key=score_dict.get)
    return recommend


def deal_mnar(df):
    """Deal with missing data with missing at random pattern."""
    recommend = 'multiple imputation'
    return recommend


def clean_missing(df, features, setting):
    """Clean missing values in the dataset.
    Parameters
    ----------
    df : DataFrame
    features : List
        List of feature names.
    Returns
    -------
    features_new : List
        List of feature names after cleaning.
    Xy_filled : array-like
        Numpy array where missing values have been cleaned.
    """

    df_preprocessed, features_new = missing_preprocess(df, features)
    if setting == 'mcar':
        recommend = deal_mcar(df_preprocessed)
    elif setting == 'mar':
        recommend = deal_mar(df_preprocessed)
    elif setting == 'mnar':
        recommend = deal_mnar(df_preprocessed)
    else:
        print("Default MAR")
        recommend = deal_mar(df_preprocessed)


    if recommend == 'mean':
        print("Applying mean imputation ...")
        Xy_filled = Imputer(missing_values=np.nan, strategy='mean').fit_transform(df_preprocessed.values)
        print("Missing values cleaned!")
    elif recommend == 'mode':
        print("Applying mode imputation ...")
        Xy_filled = Imputer(missing_values=np.nan, strategy='most_frequent').fit_transform(df_preprocessed.values)
        print("Missing values cleaned!")
    elif recommend == 'knn':
        print("Applying knn imputation ...")
        with NoStdStreams():
            Xy_filled = KNN().fit_transform(df_preprocessed.values);
        print("Missing values cleaned!")
    elif recommend == 'matrix factorization':
        print("Applying matrix factorization ...")
        with NoStdStreams():
            Xy_filled = MatrixFactorization().fit_transform(df_preprocessed.values);
        print("Missing values cleaned!")
    elif recommend == 'multiple imputation':
        print("Applying multiple imputation ...")
        with NoStdStreams():
            Xy_filled = IterativeImputer().fit_transform(df_preprocessed.values)
        print("Missing values cleaned!")
    else:
        print("Error: Approach not available!")
    return features_new, Xy_filled


def handle_missing(df, setting = 'mar', na_values=['n/a', 'na', '--', '?']):
    """Handle missing values.
    Recommend the approprate approach to the user given the missing mechanism
    of the dataset. The user can choose to adopt the recommended approach or take
    another available approach.
    For MCAR, the following methods are evaluated: 'list deletion', 'mean',
    'mode', 'k nearest neighbors', 'matrix factorization', 'multiple imputation'.
    For MAR, the following methods are evaluated: 'k nearest neighbors',
    'matrix factorization', 'multiple imputation'.
    For MNAR, 'multiple imputation' is adopted.
    Parameters
    ----------
    features : list
        List of feature names.
    Xy : array-like
        Complete numpy array (target required and not optional).
    Returns
    -------
    features_new : List
        List of feature names after cleaning.
    Xy_filled : array-like
        Numpy array where missing values have been cleaned.
    """
    flag = identify_missing(df, na_values)
    features_new = df.columns
    Xy_filled = np.asarray(df)
    if flag:
        features_new, Xy_filled = clean_missing(df.columns, df, setting)
    df_filled = pd.DataFrame(Xy_filled, columns=features_new)
    return df_filled