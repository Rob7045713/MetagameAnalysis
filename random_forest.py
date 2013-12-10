""" Implementation of a random forests algorithm

Copyright (c) 2012 Robert Argue (rargue@cs.umd.edu)
"""

__author__ = 'Rob Argue'



import random as rnd
import dectree.DTree as dt
import numpy as np
import analysis as an



class RandomForest:
    """ Random forest of decision trees which can be trained and used for
    prediction
    """

    def __init__(self, train_set, label, N_feat = 5, N_tree = 50, max_depth = 10):
        """ Initializes and trains the forest

        Arguments:
            train_set   - pandas.DataFrame containing the training data

            label       - Name of the label column in the DataFrame

            N_feat      - Number of features to consider for each trees
                          Default = 5

            N_tree      - Number of trees to use in the forest
                          Default = 50

            max_depth   - Maximum depth of trees to create
                          Default = 10        
        """

        self.train(train_set, label, N_feat, N_tree, max_depth)



    def train(self, train_set, label, N_feat = 5, N_tree = 50, max_depth = 10):
        """ Trains the forest, and stores trees for prediction

        Arguments:
            train_set   - pandas.DataFrame containing the training data

            label       - Name of the label column in the DataFrame

            N_feat      - Number of features to consider for each trees
                          Default = 5

            N_tree      - Number of trees to use in the forest
                          Default = 50

            max_depth   - Maximum depth of trees to create
                          Default = 10        
        """

        self.trees = []
        self.N_feat = N_feat
        self.N_tree = N_tree
        for i in range(N_tree):
            t_set = train_set.copy()
            t_set = reduce_features(self.N_feat, t_set, label)
            self.trees.append(dt.get_tree(t_set, label, max_depth))



    def predict(self, test_set):
        """ Predicts the label of a set of data according to trained forest

        Arguments:
            test_set - pandas.DataFrame containing the data to predict the
                       label of

        Returns:
            List of labels for the data
        """

        results = []

        for i in range(self.N_tree):
            results.append(self.trees[i].predict(test_set))
        
        results = np.mean(results, axis = 0)
        
        return results.tolist()



    def bootstrap_validate(self, test_set, label):
        """ Computes the F1 score for a test set of data

        Arguments:
            test_set    - pandas.DataFrame containing the test data

            labels      - Name of the label column in the DataFrame
            
        """

        pred = self.predict(test_set)
        true = test_set[label]
        
        return an.bootstrap_evaluate(pred, true)



def reduce_features(N_feat, data_frame, label):
    """ Reduces the number of features in a set of data through random
    selection

    Arguments:
        N_feat      - Number of features to reduce down to

        data_frame  - pandas.DataFrame containing the data

        label       - Name of the label column in the DataFrame

    Returns:
        (Mean, StandardDeviation) of the F1 scores produced by the
        bootstrapping
        
    """

    # get list of features
    feats = data_frame.columns.values
    feats = feats.tolist()
    feats.remove(label)

    # num to remove
    num = len(feats) - N_feat
    for i in range(num):
        idx = rnd.randint(0, len(feats) - 1)
        data_frame.pop(feats[idx])
        feats.remove(feats[idx])

    return data_frame