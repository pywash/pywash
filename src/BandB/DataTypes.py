import random

from datacleanbot.bayesian.bin import abda
from datacleanbot.dataclean import NoStdStreams
from scipy.io import savemat
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def infer_feature_type(feature):
    """Infer data types for the given feature using simple logic.
    Possible data types to infer: boolean, date, float, integer, string
    Feature that is not either a boolean, a date, a float or an integer,
    is classified as a string.
    Parameters
    ----------
    feature : array-like
        A feature/attribute vector.
    Returns
    -------
    data_type : string
        The data type of the given feature/attribute.
    """
    data_type = ""
    types = ["date", "float64", "int64", "string"]
    weights = [0, 0, 0, 0]  # Weights corresponding to the data types
    feature_len = len(feature)

    indices_number = int(0.1 * feature_len)  # Number of different values to check in a feature
    indices = random.sample(range(0, feature_len), min(indices_number, feature_len))  # Array of random indices

    # If the feature only contains two different unique values, then infer it as boolean
    if len(pd.unique(feature)) == 2:
        data_type = "bool"
    else:
        for i in indices:
            try:
                if (len(feature[i]) <= 10
                        and (((feature[i][2:3] == '-' or feature[i][2:3] == '/')
                              and (feature[i][5:6] == '-' or feature[i][5:6] == '/'))
                             or ((feature[i][4:5] == '-' or feature[i][4:5] == '/')
                                 and (feature[i][7:8] == '-' or feature[i][7:8] == '/')))):
                    weights[0] += 1  # Date
                else:
                    weights[3] += 1  # String
            except (TypeError, ValueError, IndexError):
                try:
                    int(feature[i])  # numeric
                    if ('.' in str(feature[i])):
                        if isinstance(feature[i], np.float64):
                            if feature[i].is_integer():
                                weights[2] += 1  # Integer
                            else:
                                weights[1] += 1  # Float
                        else:
                            weights[1] += 1  # Float
                    else:
                        weights[2] += 1  # Integer
                except (TypeError, ValueError, IndexError):
                    weights[3] += 1  # String
        # For debugging
        # print ("Date: {}, Float64: {}, Int64: {},
        # String: {}".format(weights[0],weights[1],weights[2],weights[3]))
        data_type = types[weights.index(max(weights))]

    return data_type


def discover_type_heuristic(data):
    """Infer data types for each feature using simple logic
    Parameters
    ----------
    data : numpy array or dataframe
        Numeric data needs to be 64 bit.
    Returns
    -------
    result : list
        List of data types.
    """
    #     df = pd.DataFrame(data)
    #     print(df)
    result = []
    if isinstance(data, np.ndarray):
        # convert float32 to float64
        data = np.array(data, dtype='float64')
        df = pd.DataFrame(data)
    else:
        df = data

    for column in df.columns:
        # print ("Trying to automatically infer the data type of the",column,"feature...") #For debugging purposes
        type_inferred = infer_feature_type(df[column])
        result.append(type_inferred)
        # print ("Result:",inferredType) #For debugging purposes
    return result


def generate_mat(Xy, extra_cardinality=1):
    """Convert data to mat format.
    In order to use the Bayesian model, data need to be converted
    to the .mat format.
    """
    data = Xy
    simple_types = discover_type_heuristic(data)
    # map simple types to meta types
    # 1: real (w positive: all real | positive | interval)
    # 2: real (w/o positive: all real | interval)
    # 3: binary data
    # 4: discrete (non-binary: categorical | ordinal | count)
    # note: for now, the implemented bayesian method version by isabel can distinguish between real, postive real, categorical and count
    # the binary data should be mapped to meta type 4 discrete instead of meta type 3 due to the limited implemented version. This may change
    # if the extended version has been implemented by isabel.
    meta_types = []
    for i in range(len(simple_types)):
        #         print(simple_types[i])
        if simple_types[i] == "bool":
            meta_types.append(4)  # may change in the future
        elif simple_types[i] == "int64" or simple_types[i] == "float64":
            if (len(np.unique(data[:, i])) < 0.02 * len(data[:, i]) and \
                    len(np.unique(data[:, i])) < 50):
                meta_types.append(4)
            else:
                if (data[:, i] > 0).all():
                    meta_types.append(1)
                else:
                    meta_types.append(2)
        else:
            meta_types.append(1)
    discrete_cardinality = []  # max for discrete feature, 1 for others
    for i in range(len(meta_types)):
        if (meta_types[i] == 4):
            discrete_cardinality.append(int(np.max(data[:, i])) + extra_cardinality)
        else:
            discrete_cardinality.append(1)
    data_dict = {'X': data,
                 'T': np.asarray(meta_types),
                 'R': np.asarray(discrete_cardinality)}
    # pprint.pprint(data_dict)
    savemat('data.mat', data_dict, oned_as='row')


def discover_type_bayesian(Xy):
    """Infer data types for each feature using Bayesian model.
    Retrieve the key with the higher value from 'weights' of the output of
    the Bayesian model. The retrieved key is the statisical type of the
    corresponding feature.
    Parameters
    ----------
    Xy : numpy array
        Xy can only be numeric in order to run the Bayesian model.
    Returns
    -------
    result : list
        List of data types.
    """
    statistical_types = []
    generate_mat(Xy)
    #     with HiddenPrints():
    with NoStdStreams():
        print("This will not be printed")
        weights = abda.main(seed=1337, dataset='data.mat', exp_id=None, args_output='./exp/temp/', args_miss=None,
                            verbose=1,
                            args_col_split_threshold=0.8, args_min_inst_slice=500, args_leaf_type='pm',
                            args_type_param_map='spicky-prior-1', args_param_init='default',
                            args_param_weight_init='uniform',
                            args_n_iters=5, args_burn_in=4000, args_w_unif_prior=100, args_save_samples=1,
                            args_ll_history=1, args_omega_prior='uniform', args_plot_iter=10, args_omega_unif_prior=10,
                            args_leaf_omega_unif_prior=0.1, args_cat_unif_prior=1);
    for i in range(len(weights)):
        #         print(max(weights[i], key=weights[i].get))
        statistical_types.append(str(max(weights[i], key=weights[i].get)))
    return statistical_types


def discover_types(Xy):
    """Discover types for numpy array.
    Both simple logic rules and Bayesian methods are applied.
    Bayesian methods can only be applied if Xy are numeric.
    Parameters
    ----------
    Xy : numpy array or DataFrame
        Xy can only be numeric in order to run the Bayesian model.
    """
    if isinstance(Xy, pd.DataFrame):
        Xy = Xy.values

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    Xy = imp_mean.fit_transform(Xy)

    print(discover_type_heuristic(Xy))
    try:
        print(discover_type_bayesian(Xy))
    except:
        print("Failed to run the Bayesian model.")