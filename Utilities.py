#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Date : 2018-06-24 to 2018-07-20
Author: Benoît Boucher

Credits: 
    Some functions were coded by myself, some others are adaptations of work from colleagues or 
    found on the web. Credit is given at the end of the appropriate function's doc-string.
"""

import os, time, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from scipy.stats import uniform


def LoadGzFormat(path_to_name):
    """
    This function combines all the data from the .pygz format in the path folder.
    This is a home made function that only works on these home made .pygz formats.
    :param
        path_to_name: str
            path to name of the folder in which all the .pygz files exists

    :return
        full: pandas.DataFrame.Object
            dataframe will all the metadata and full_stat_vector for the original db 
            without all the proximity object.

    Credit: Based on Alexandre Désilet-Benoit's work, CONTXTFUL
    """
    try:
        path = os.path.abspath(path_to_name)
        name = os.path.basename(path)
        path += "/"

        metadata = []
        places = list(os.walk(path))
        largest = 0
        for things in places[0][2]:
            if "_meta.pygz" in things:
                meta = pd.read_pickle(path+name + "_meta.pygz",'gzip')
            else:
                this = things.replace(name+"_","")
                this = int(this.replace("_fsv.pygz",""))
                if this > largest:
                    largest = this
        stat = pd.read_pickle(path+name + "_1000_fsv.pygz",'gzip')
        j = 2000
        while j < largest:
            stat = pd.concat([stat,
                pd.read_pickle(path+name + "_"+str(j)+"_fsv.pygz",'gzip')
                ])
            j += 1000
        stat = pd.concat([stat,
                pd.read_pickle(path+name + "_"+str(largest)+"_fsv.pygz",'gzip')
                ])
        full = meta
        full = full.assign(full_stat_vector=stat.full_stat_vector)
        return full
    except:
        print("Either path is not well written or the folders are not present")
        return False


def LabelClusterMatrix(df, y_predict, label_column="model", percentage=True):
    """
    This function takes a reference DataFrame that contains true labels and compares them with 
    predicted labels from a clustering algorithm. It returns a matrix where the rows are 
    the true labels, the columns are the new clusters, and each item(i,j) in the matrix is the 
    number of samples that has label(i) and was classified into cluster(j).

    :param
        df: pandas.DataFrame that contains the target data. The target data are categories
            written in string format. NOTE: df HAS to be a dataframe. The groupby does not work
            if df is a series (ie: use df, not df.model)
        y_predict: a numpy.array with shape (n,) where n == len(df.label_column). It contains the
            results of the clustering algorithm, which is a number associated with a cluster for
            each sample
        label_column: a string which corresponds to the name of the df column which contains the
            target data. We assume that the input df might contain more than 1 column and this
            tells us which column is relevant. Even if df contains only 1 column, we still
            need the name to get the groupby method going.
        percentage: boolean value. If False, the values returned are the raw number of labels.
            If true, the values returned are a percentage (between 0-1) between the number of
            labels in the cluster and the total number of labels. Each label will sum to 1
            across all clusters.

    :return
        lcm: [label-cluster-matrix] a pandas.DataFrame object in which the
            index is the labels and the columns are the clusters.
            Each datapoint is the total number of sample with
            a particular label grouped into the corresponding cluster. 

    Tip: Use the output of LoadGzFormat as df. Then use the output of this function as input 
        for HistoCluster or ClusterScore
    """
    
    clusters = set(y_predict)
    y, labels = pd.factorize(df.loc[:, label_column])
    lcm = pd.DataFrame(index=labels, columns=clusters)

    if percentage:
        divider = df.groupby(label_column).size()
    else:
        divider = 1

    for cluster in clusters:
        lcm.loc[:,cluster] = df[y_predict == cluster].groupby(label_column).size() / divider
    
    lcm = lcm.fillna(0)

    return lcm


def HistoCluster(lcm):
    """
    This function creates a figure with as many subplots as there are clusters.
    Each subplot is a histogram that has as many bins as there are labels.
    This is useful for showing the result of clustering labeled data to see in which
    cluster each labeled sample ended up.
    :param
        lcm: [label-cluster-matrix] a pandas.DataFrame object in which the 
            index is the labels and the columns are the clusters. 
            Each datapoint is the total number of sample with 
            a particular label grouped into the corresponding cluster.
    :return None; shows a figure

    Tip: use the output of LabelClusterMatrix as input for HistoCluster
    """
    n_labels, n_clusters = lcm.shape
    labels = lcm.index
    clusters = lcm.columns
    
    max_rows = 4
    n_rows = min(n_clusters, max_rows)
    n_cols = math.ceil(n_clusters/max_rows)

    fig,ax=plt.subplots(nrows=n_rows, ncols=n_cols, sharex="all", sharey="row", squeeze=False)    

    colors = []
    for c in range(n_labels):
        colors.append(plt.cm.tab20(c))

    width = 0.75 # the width of the bars 
    ind = np.arange(n_labels)  # the labels locations of each class
    
    for c in range(n_clusters):
        row = c%n_rows
        col = c//n_rows
        ax[row,col].barh(ind, lcm.iloc[:,c], width, align="center", color=colors)
        ax[row,col].set_yticks(ind)
        ax[row,col].set_yticklabels(labels, minor=False, size="xx-small")
        ax[row,col].set_title('Cluster# ' + str(clusters[c]), size="x-small", va="top")
    
    plt.show()

def ClusterScore(lcm):
    """
    This is a homemade metric that measures the quality of a clustering result.
    It calculates its score based on the result of LabelClusterMatrix function
    with the "percentage=True" parameter.
    Every entry in the lcm will be passed into the "abso" function.
    A good score is attributed if the ratio is either close to 1 or 0, signifying
    that the labels were indeed clustered into clusters in an exclusive fashion.
    In other words, a specific label should be found inside a unique cluster and
    nowhere else.

    :param lcm: use the output of LabelClusterMatrix with parameter "percentage=True"
    :return a single float between 0-1 representing the score; 1 being the best.
    """
    if ((lcm > 1).any() | (lcm < 0).any()).any():
        print("Error. Detected 1 or more values outside of 0 <= x <= 1")
        return -1
    else:
        matrix_score = lcm.apply(abso)
        return matrix_score.mean().mean()

def quad(x):
    """
    This applies a quadratic function to x.
    It has been calibrated so that quad(0) == quad(1) == 1
    and quad(0.5) == 0
    Its goal is to give a score based on how close to 0 or 1 a number is.
    :param x: a float
    :return a float transformed by the quadratic equation below
    """
    return 4*x**2 - 4*x + 1

def abso(x):
    """
    This applies an absolute function to x.
    It has been calibrated so that abso(0) == abso(1) == 1
    and abso(0.5) == 0
    Its goal is to give a score based on how close to 0 or 1 a number is.
    :param x: a float
    :return: a float transformed by the absolute equation below
    """
    return 2*abs(x-0.5)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    :param
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    :return a matplotlib.pyplot plot object

    Credit: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def class_score(y_predict, y_true):
    """
    This function returns a breakdown of the accuracy score for each class in a prediction model 
    instead of the overall score returned by sklearn.model.score functions.
    
    :param
        y_predict: a list or numpy.array containing the class predictions from your model
        y_true: a list of numpy array containing the reference (true) labels of your data

    :return
        scores: a list of length (n_labels) that contains the accuracy of each label independently
    """
    scores = []
    for label in set(y_true):
        scores.append( list(y_predict[y_true == label]).count(label) / list(y_true).count(label) )
    return scores

class log_uniform():
    """
    A log-uniform continuous random variable
    It is a homebrew function in the same vein as the scipy.stats functions.

    :param
        a: the exponent of the beginning of the range. default = -1
        b: the exponent of the end of range. default = 0
        base: The base of the exponents. Default=10

    :return
        a distribution of probability just like scipy.stats.distributions,
        which has a useful rvs method for sampling variables.
        The default range is between 0.1 and 1.

    How to use:
        log_uniform(a=2, b=3).rvs(size=10) 
        returns an array of 10 random numbers between 100 and 1000

    Note: Unlike the true scipy.stats.distributions functions, you have to pass
        a and b in the class; you cannot pass them in the rvs method.

    Credit: marty, as seen on https://stackoverflow.com/a/49538913 
    """

    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b-a
        self.base = base

    def rvs(self, size=None, random_state=None):
        """
        This method generates random numbers sampled from the log_uniform distribution.

        :param
            size: an Integer that tells how many samples to pull out.
            random_state: an Integer that seeds the RNG.

        :return
            np.array that contains "size" random values pulled from the log_uniform distribution.
            Default size is None, so it's a single float instead of an array
        """
        uniform_distrib = uniform(loc=self.loc, scale=self.scale)
        return np.power(self.base, uniform_distrib.rvs(size=size, random_state=random_state))


if __name__ == "__main__":
    t0 = time.time()
    df = LoadGzFormat("/home/bboucher/workspace/DATA/gzDataFormat/amt_confirmed_data")
    #df = LoadGzFormat("/home/bboucher/workspace/DATA/gzDataFormat/lifeofpix_analyzed_data")
    t1 = time.time()
    print(round(1000000*(t1-t0)/len(df)),"microseconds per element in df")

