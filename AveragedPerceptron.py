""" Implementation of the averaged perceptron algorithm

Copyright (c) 2012 Robert Argue (rargue@cs.umd.edu)
"""

__author__ = 'Rob Argue'



import numpy as np



class AveragedPerceptron:
    """ Averaged perceptron which can be trained and used for prediction
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

        w = [0.0] * num_feats
        u = [0.0] * num_feats
        b = 0.0
        beta = 0.0
        c = 1.0

        for i in range(MaxIter):
        
            for e in Data.index:
                y = Data.ix[e].values[-1]
                x =  Data.ix[e].values[:-1]
                a = np.dot(w, x) + b
        
                if (y * a <= 0):
                    w = w + y * x
                    b = b + y
                    u = u + y *c * x
                    beta = beta + y * c
        
                c = c + 1.0

        # store the computed weights and bias
        self.w = w - (1.0 / c) * u
        self.b = b - (1.0 / c) * beta



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