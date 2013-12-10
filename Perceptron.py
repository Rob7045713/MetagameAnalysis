""" Implementation of the perceptron algorithm

Copyright (c) 2012 Robert Argue (rargue@cs.umd.edu)
"""

__author__ = 'Rob Argue'



import numpy as np



class Perceptron:
    """ Perceptron which can be trained and used for prediction
    """

    def __init__(self, Data, MaxIter):
        """ Initializes and trains the perceptron

        Arguments:
            Data    - pandas.DataFrame containing the training data

            MaxIter - Maximum number of iterations to train the perceptron
                      for
        """

        Data2 = Data.dropna()
        self.train(Data2, MaxIter)



    def train(self, Data, MaxIter):
        """ Trains the perceptron, and stores the weights and bias

        Arguments:
            Data    - pandas.DataFrame containing the training data

            MaxIter - Maximum number of iterations to train the perceptron
                      for
        """

        num_feats = len(Data.columns) - 1

        self.w = [0.0] * num_feats
        self.b = 0.0

        for i in range(MaxIter):
            for e in Data.index:
                y = Data.ix[e].values[-1]
                x =  Data.ix[e].values[:-1]
                a = np.dot(self.w, x) + self.b
                if (y * a <= 0):
                    self.w = self.w + y * x
                    self.b = self.b + y



    def predict(self, Data, label = None):
        """ Predicts the label of a set of data according to trained weights
        and bias

        Arguments:
            Data    - pandas.DataFrame containing the data to predict the
                      label of

            label   - Name of the label column in the DataFrame (if
                      applicable)
                      Default = None (no label column in the DataFrame)

        Returns:
            List of positive / negative labels for the data
        """
        
        Data2 = Data.copy()

        if (label != None):
            Data2.pop(label)

        a = np.dot(Data2, self.w) + self.b

        return np.sign(a)