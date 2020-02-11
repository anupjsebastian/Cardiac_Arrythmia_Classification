import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import stumpy


def rename_cols(df):
    """
    Renames the columns of the DataFrame as its column numbers.

    Parameters:
    df: The DataFrame which needs its columns to be renamed.

    """
    df.columns = list(range(len(df.columns)))

def plot_series(df, n = None, figsize = (20, 7)):
    sns.set()
    """
    Plots time series data in a single plot.

    Each row of the DataFrame df is considered as a different time series.


    Parameters:
    df: The DataFrame to plot the data from.

    n: the number of rows to plot. If no value is provided, it plots the entire DataFrame.

    figsize: A tuple which represents the size of the plot as (length, height)

    """
    if n == None:
        n = len(df)
    plt.figure(1, figsize = figsize)
    for i in range(n):
        sns.lineplot(x = np.linspace(0, 1.5, num = len(df.iloc[i])), y = df.iloc[i],  data = df);
    plt.xlabel('Time(s)')
    plt.ylabel('Signal Strength')


def random_plotter(df, figsize = (15, 7)):
    """
    Creates a plot selecting two random columns from a DataFrame against
    each other, with the target variable colored

    Parameters:
    df: The DataFrame to plot from

    """
    df2 = df.copy()
    df2[187] = df[187].astype(object)
    df2[187] = df2[187].map({0:'0', 1: '1', 2: '2', 3: '3', 4: '4'})
    df2.rename(columns = {187: 'Target'}, inplace = True)

    x_axis = np.random.choice(range(187))
    y_axis = np.random.choice(range(187))
    while x_axis == y_axis:
        y_axis = np.random.choice(range(187))
    plt.figure(1, figsize = figsize)
    sns.scatterplot(x = x_axis, y = y_axis, hue = 'Target', data = df2, palette=sns.color_palette("Set1", df2.Target.nunique()))
    plt.xlabel('Column '+ str(x_axis))
    plt.ylabel('Column '+ str(y_axis))

def explained_variance_ratio_plot(cumulative_variance, threshold = 0.9):
    """
    Creates a plot for the number of features vs
    the cumulative explained variance ratio

    Parameters:
    cumulative_variance: an array or vector of the cumulative sum of the variance ratio.
    Can be calculated using PCA.explained_variance_ratio_.cumsum()

    threshold: Value from 0 to 1. Threshold value for the explained variance ratio
    for which you want to see the number of features required. 0.9 by default

    """
    thresh = np.argwhere(np.array(cumulative_variance) >= threshold)[0][0]
    print('90% Information explained at', thresh, 'features.')
    print('=' * 140)

    plt.figure(1, figsize = (15, 7))
    sns.lineplot(x = np.arange(len(cumulative_variance)), y = cumulative_variance)
    plt.vlines(x = thresh, ymin = np.min(cumulative_variance), ymax = 1)
    plt.xlabel('Number of Features')
    plt.ylabel('Explained Variance Ratio');

def print_metric_results(y_train, y_test, train_preds, test_preds):
    """
    Calcualtes and prints the Accuracy Score and F1 Score for both the
    training and test sets.

    Parameters:
    y_train: the actual labels of the training data.

    y_test: the actual labels of the test data.

    train_preds: the predicted labels of the training data.

    test_preds: the predicted labels of the test data.

    """
    print('Train Accuracy:', accuracy_score(y_train, train_preds))
    print('Train F1 Score:', f1_score(y_train, train_preds, average = 'weighted'))

    print('=' * 40)

    print('Test Accuracy:', accuracy_score(y_test, test_preds))
    print('Test F1 Score:', f1_score(y_test, test_preds, average = 'weighted'))



def plot_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    # Taken from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    plt.figure(1, figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def create_stump_df(df, m):
    """
    Helper function. Creates an empty DataFrame with the correct number of
    columns when the matrix profile is calculated.

    Parameters:
    df: The DataFrame which needs to be stumpified.

    m: The subsequence length for the Matrix Profile calculation.

    Returns:
    df_stump: An empty DataFrame with the correct number of columns for the stumpified matrix.

    """
    mp = stumpy.stump(df.iloc[0, :-1].values, m)
    df_stump = pd.DataFrame(columns = list(range(len(mp))))
    return df_stump


def stumpify(df, m):

    """
    Calculates the Matrix profile for an entire DataFrame. Use caution as it takes
    time to run.

    Parameters:
    df: The DataFrame which needs to be stumpified.

    m: The subsequence length for the Matrix Profile calculation.

    Returns:
    df_stump: A DataFrame containing the Matrix Profile of the time series' in the
    input DataFrame.

    """

    df_stump = create_stump_df(df, m)
    for i in range(len(df)):
        mp = stumpy.stump(df.iloc[i, :].values, m)
        df_stump = df_stump.append(pd.Series(mp[:, 0]), ignore_index=True)
    return df_stump
