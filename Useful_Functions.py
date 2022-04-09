import os
import json
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def industry_identifier(code,NAICS_DF):
    industry_name = 'Undefined'
    first_two_digits = int(str(code)[:2])
    if first_two_digits in NAICS_DF.index.values:
        if code != 0:
           industry_name = NAICS_DF.loc[first_two_digits, 'IndustryName']

    return industry_name

def merge_on_date(df_A, df_B, df_A_column, df_B_column):
    for index, row in df_B.iterrows():
        start_date = row['start date']
        end_date = row['end date']
        for index1, row1 in df_A.iterrows():
            if row1['ApprovalDate'] < end_date and row1['ApprovalDate'] >= start_date:
                df_A.at[index1, df_A_column] = row[df_B_column]

    return df_A

def fix_year(series_A, series_B):
    empty_list = []
    for index, value in enumerate(series_A):
        k = value.replace(year = int(series_B[index]))
        empty_list.append(k)
    series_A = empty_list
    return series_A

def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        return x.replace('$', '').replace(',', '')
    return x


def precision(y_true, y_pred):
    """
    Returns the precision score.

    Parameters:
        y_true (1D numpy array): 1D array of true binary labels
                                 np.array([0, 1, 0, 0, ..]).
        y_pred (1D numpy array): 1D array of predicted binary labels
                                 np.array([1, 0, 0, 1, ..]).

    Returns:
        (float): The precision.
    """

    # Write your code here
    try:
        return count_true_positives(y_true, y_pred) / (
                    count_true_positives(y_true, y_pred) + count_false_positives(y_true, y_pred))
    except ZeroDivisionError:
        return 0

def accuracy(y_true, y_pred):

    """
    Copied from Assignment 8
    Returns a accuracy score for two 1D numpy array of the same length.

    Parameters:
        y_true (1D numpy array): 1D array of true binary labels
                                 np.array([0, 1, 0, 0, ..]).
        y_pred (1D numpy array): 1D array of predicted binary labels
                                 np.array([1, 0, 0, 1, ..]).

    Returns:
        (float): Accuracy score of y_true and y_pred.
    """

    # Write your code here
    TP_plus_TN = 0
    for i in range(len(y_true)):
        if y_true[i] - y_pred[i] == 0:
            TP_plus_TN +=1

    accuracy_score = TP_plus_TN/len(y_true)
    return accuracy_score

def recall(y_true, y_pred):
    """
    Returns the recall score.

    Parameters:
        y_true (1D numpy array): 1D array of true binary labels
                                 np.array([0, 1, 0, 0, ..]).
        y_pred (1D numpy array): 1D array of predicted binary labels
                                 np.array([1, 0, 0, 1, ..]).

    Returns:
        (float): The recall.
    """

    # Write your code here
    recall_score = count_true_positives(y_true, y_pred) / (count_true_positives(y_true, y_pred) + count_false_negatives(y_true, y_pred))

    return recall_score


def false_alarm_rate(y_true, y_pred):
    """
    Returns the false alarm rate.

    Parameters:
        y_true (1D numpy array): 1D array of true binary labels
                                 np.array([0, 1, 0, 0, ..]).
        y_pred (1D numpy array): 1D array of predicted binary labels
                                 np.array([1, 0, 0, 1, ..]).

    Returns:
        (float): The false positive rate.
    """

    # Write your code here
    false_alarm_score = count_false_positives(y_true, y_pred) / (count_false_positives(y_true, y_pred) + count_true_negatives(y_true, y_pred))

    return false_alarm_score


def f_beta(y_true, y_pred, beta):
    """
    Returns the F-beta score.

    Parameters:
        y_true (1D numpy array): 1D array of true binary labels
                                 np.array([0, 1, 0, 0, ..]).
        y_pred (1D numpy array): 1D array of predicted binary labels
                                 np.array([1, 0, 0, 1, ..]).
        beta (float): The beta parameter for the F-beta metric.

    Returns:
        (float): The F-beta score.
    """


    try:
        return (1 + beta ** 2) * precision(y_true, y_pred) * recall(y_true, y_pred) / (beta ** 2 * precision(y_true, y_pred) + recall(y_true, y_pred))
    except ZeroDivisionError:
        return 0


def count_false_positives(y_true, y_pred):
    """
    Returns the number of false positives.

    Parameters:
        y_true (1D numpy array): 1D array of true binary labels
                                 np.array([0, 1, 0, 0, ..]).
        y_pred (1D numpy array): 1D array of predicted binary labels
                                 np.array([1, 0, 0, 1, ..]).

    Returns:
        (int): The number of false positives detected.
    """

    FP = 0
    for i in range(len(y_true)):
        if y_true[i] - y_pred[i] == -1:
            if y_true[i] == 0:
                FP += 1

    return FP


def count_false_negatives(y_true, y_pred):
    """
    Returns the number of false negatives.

    Parameters:
        y_true (1D numpy array): 1D array of true binary labels
                                 np.array([0, 1, 0, 0, ..]).
        y_pred (1D numpy array): 1D array of predicted binary labels
                                 np.array([1, 0, 0, 1, ..]).

    Returns:
        (int): The number of false negatives detected.
    """

    FN = 0
    for i in range(len(y_true)):
        if (y_true[i] - y_pred[i] == 1) and (y_true[i] == 1):
            FN += 1

    return FN


def count_true_positives(y_true, y_pred):
    """
    Returns the number of true positives.

    Parameters:
        y_true (1D numpy array): 1D array of true binary labels
                                 np.array([0, 1, 0, 0, ..]).
        y_pred (1D numpy array): 1D array of predicted binary labels
                                 np.array([1, 0, 0, 1, ..]).

    Returns:
        (int): The number of true positives detected.
    """

    # Write your code here
    TP = 0
    for i in range(len(y_true)):
        if (y_true[i] - y_pred[i] == 0) and (y_true[i] == 1):
            TP += 1

    return TP


def count_true_negatives(y_true, y_pred):
    """
    Returns the number of false negatives.

    Parameters:
        y_true (1D numpy array): 1D array of true binary labels
                                 np.array([0, 1, 0, 0, ..]).
        y_pred (1D numpy array): 1D array of predicted binary labels
                                 np.array([1, 0, 0, 1, ..]).

    Returns:
        (int): The number of true negatives detected.
    """

    # Write your code here
    TN = 0
    for i in range(len(y_true)):
        if (y_true[i] - y_pred[i] == 0) and (y_true[i] == 0):
            TN += 1

    return TN

def word_detector(words, texts):

    """
    Copied from Assignment 8
    Returns a DataFrame with detections of words.

    Parameters:
        words (list): A list of words to look for.
        texts (Series): A series of strings to search in.

    Returns:
        (DataFrame): A DataFrame with len(words) columns and texts.shape[0] rows.
    """


    row_to_df = []
    for index, value in texts.items():
        row = [0]*len(words)
        for y in range(len(words)):
            n = 0
            if words[y] in value.lower():
                n = 1
            row[y] = n
        row_to_df.append(row)
    df = pd.DataFrame(row_to_df,columns=words)
    df = df.set_index(texts.index)
    return df