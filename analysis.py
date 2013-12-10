""" Module providing some statistics utility methods

Copyright (c) 2012 Robert Argue (rargue@cs.umd.edu)
"""

__author__ = 'Rob Argue'



import random as rnd
import numpy as np



def accuracy(pred, true):
    """ Compute the accuracy of a set of predictions as (TotalCorrect / Total)

    Arguments:
        pred - List of predicted values

        true - List of actual values

    Returns:
        Accuracy of the predictions
    """

    return ((float) (sum(pred == true)) / len(pred))



def recall(pred,true):
    """ Compute the recall of a set of predictions as 
    (TruePositive / (TruePositive + FalseNegative))

    Arguments:
        pred - List of predicted values

        true - List of actual values

    Returns:
        Recall of the predictions
    """

    return np.mean(np.array([pred[i]==true[i] for i in range(len(true)) if true[i]==1]))



def precision(pred, true):
    """ Compute the precision of a set of predictions as 
    (TruePositive / (TruePositive + FalsePositive))

    Arguments:
        pred - List of predicted values

        true - List of actual values

    Returns:
        Precision of the predictions
    """
    return np.mean(np.array([pred[i]==true[i] for i in range(len(true)) if pred[i]==1]))



def f1_score(pred, true):
    """ Compute the F1 score of a set of predictions as 
    (2 * precision * recall / (precision + recall))

    Arguments:
        pred - List of predicted values

        true - List of actual values

    Returns:
        F1 score of the predictions
    """
    p = precision(pred, true)
    r = recall(pred, true)
    return (2 * p * r) / (p + r)



def bootstrap_evaluate(pred, true, num_folds = 10):
    """ Compute the F1 score of a set of predictions using case resampling

    Arguments:
        pred        - List of predicted values

        true        - List of actual values

        num_folds   - Number of different random samplings to do
                      Default = 10

    Returns:
        (Mean, StandardDeviation) of the F1 scores produced by the
        bootstrapping
    """

    N = len(pred)
    scores = []
    
    for k in range(num_folds):
        true_k = []
        pred_k = []
    
        # Note that the same indices must be used for both sets
        for n in range(N):
            idx = rnd.randint(0, N - 1)
            true_k.append(true[idx])
            pred_k.append(pred[idx])
    
        scores.append(f1_score(pred_k, true_k))
    
    return np.mean(scores), np.std(scores)



def bootstrap(items):
    """ Create a new random set from the given set by selecting random values
    with replacement

    Arguments:
        items - List of items to use for sampling

    Returns:
        A new list of items the same size as the original sampled from the 
        original data
    """

    N = len(items)
    boot = []

    for n in range(N):
        idx = rnd.randint(0, N - 1)
        boot.append(items[idx])

    return boot