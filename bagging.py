""" Implementation of bagging (bootstrap aggregating) for decision trees

Copyright (c) 2012 Robert Argue (rargue@cs.umd.edu)
"""

__author__ = 'Rob Argue'



import analysis as an
import dectree.DTree as dt
import numpy as np



class Bagger:
    """ Bootstrap aggregator for decision trees
    """

    def __init__(self):
        """ Initializes the bagger to be empty 
        """

        self.M = 0
        self.trees = []



    def train(self, train_set, label, M = 10):
        """ Trains the bagger and stores the resulting trees

        Arguments:
            train_set   - pandas.DataFrame containing the training data

            label       - Name of the label column in the DataFrame

            M           - Number of trees to generate
                          Default = 10
        """

        N = train_set.shape[0]
        self.M = M
        self.trees = []

        for i in range(M):
            idxs = an.bootstrap(range(N))
            b_set = train_set.ix[idxs]
            b_set.index = np.arange(N)
            self.trees.append(dt.get_tree(b_set, label))



    def evaluate(self, test_set):
        """ Evaluates a test set based on the trained trees

        Arguments:
            test_set - pandas.DataFrame containing the data to evaluate

        Returns:
            List of predicted positive / negative labels
        """

        results = []

        for i in range(self.M):
            results.append(self.trees[i].predict(test_set))

        results = np.sign(np.sum(results, axis = 0))

        return results.tolist()



    def bootstrap_validate(self, test_set, label):
        """ Computes the F1 score for a test set of data

        Arguments:
            test_set    - pandas.DataFrame containing the test data

            labels      - Name of the label column in the DataFrame
            
        Returns:
            (Mean, StandardDeviation) of the F1 scores produced by the
            bootstrapping
        """

        pred = self.evaluate(test_set)
        true = test_set[label]

        return an.bootstrap_evaluate(pred, true)