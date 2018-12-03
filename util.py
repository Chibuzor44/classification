import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



def non_numerical_col(df):
    """
    This function takes a dataframe and returns a list of column names with
    string values
    :param df: dataframe
    :return: list of column names
    """
    ls = []
    for col in df.columns:
        if type(df[col][0]) == str:
            ls.append(col)
    return ls



def metric(y_true, y_pred):
    """
    Calculates and returns precision, recall, accuracy, matrix
    :param y_true: true class label numpy array
    :param y_pred: predicted classes numpy array
    :return: precision, recall, accuracy, matrix
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    matrix = np.array([[tp, fp], [fn, tn]])
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    accuracy = (tp + tn)/(tn+fp+fn+tp)
    return precision, recall, accuracy, matrix



def regex(df, columns = []):
    """
    extracts all digits in a column within a string and converts to floats
    :param df: dataframe
    :param columns: list of columns to convert values to floats
    :return: None
    """
    for col in columns:
        df[col] =  df[col].str.extract(r"([-\.0-9]+)").astype(float)



def plot_cols(dct, name="data"):
    """
    Plots a histogram from a dictionary
    :param dct: dictionary
    :param name: name of column
    :return: None
    """
    x_axis = np.arange(len(dct))
    plt.figure(figsize=(9, 5))
    plt.bar(x_axis, dct.values(), align='center')
    plt.xticks(x_axis, dct.keys())
    plt.title("Distribution of data in {}".format(name))



def restructure_cols(df):
    category_dict = {"x35": {"monday": "mon", "tuesday": "tue", "wednesday": "wed",
                             "thurday": "thur", "friday": "fri"},
                     "x68": {"January": "Jun", "July": "Jul", "sept.": "Sept", "Dev": "Dec"}}

    df.replace(category_dict, inplace=True)


def flter_columns(df):
    cols = ['x37', 'x58', 'x35_mon', 'x75', 'x45', 'x68_Feb', 'x34_tesla',
           'x35_tue', 'x68_Jul', 'x97', 'x83', 'x41', 'x99', 'x44', 'x96', 'x72',
           'x21', 'x34_Honda', 'x68_Apr', 'x70', 'x5', 'x66', 'x10', 'x50', 'x40',
           'x73', 'x63', 'x53', 'x22', 'x51', 'x2', 'x1', 'x12', 'x4', 'x56',
           'x78', 'x79', 'x27', 'x43', 'x95', 'x57', 'x33', 'x34_Toyota', 'x0',
           'x19', 'x8', 'x3', 'x69', 'x42', 'x25', 'y']

    try:
        return df[cols]
    except:
        return df[cols[:-1]]



def load_data(path):
    """
    Loads in the data, and performs all cleaning processes
    :param path: path to data file
    :return: dataframe
    """
    df = pd.read_csv(path)
    regex(df, ["x41", "x45"])
    restructure_cols(df)
    df = pd.get_dummies(df, prefix=["x34","x35","x68","x93"])
    df = flter_columns(df)
    df.dropna(inplace=True)
    return df