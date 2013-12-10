""" Implementation of the kernelized perceptron algorithm

Copyright (c) 2012 Robert Argue (rargue@cs.umd.edu)
"""

__author__ = 'Rob Argue'



import numpy as np



class KernelizedPerceptron:
    """ Kernelized perceptron which can be trained and used for prediction
    """

    def __init__(self, Data, label, MaxIter, deg = 1):
        """ Initializes and trains the perceptron

        Arguments:
            Data    - pandas.DataFrame containing the training data

            label   - Name of the label column in the DataFrame

            MaxIter - Maximum number of iterations to train the perceptron
                      for

            deg     - Degree of the polynomial kernel function
                      Default = 1 (linear)
        """

        Data2 = Data.dropna()
        self.train(Data2, label, MaxIter, deg)

    

    def train(self, Data, label, MaxIter, deg = 1):
        """ Trains the perceptron, and stores all information for prediction

        Arguments:
            Data    - pandas.DataFrame containing the training data

            label   - Name of the label column in the DataFrame

            MaxIter - Maximum number of iterations to train the perceptron
                      for

            deg     - Degree of the polynomial kernel function
                      Default = 1 (linear)
        """

        X = Data.as_matrix(Data.columns[:-1])
        y = Data[label].values

        num_examples, num_feats = X.shape

        # save some things
        self.num_examples = num_examples
        self.deg = deg
        self.X = X
        self.y = y
        self.alpha = np.zeros(num_examples, dtype=np.float64)
        self.b = 0.0

        # some precomputation
        kern = np.zeros((num_examples, num_examples))
        for i in range(num_examples):
            for j in range(num_examples):
                kern[i,j] = Kpoly(X[i], X[j], deg)

        for i in range(MaxIter):
            for n in range(num_examples):
                a = np.sum(kern[:,n] * self.alpha * y) + self.b

                if (y[n] * a <= 0):
                    self.alpha[n] += 1.0
                    self.b += y[n]

    

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

        X = Data2.as_matrix()

        kern = np.zeros((self.num_examples, X.shape[0]))
        for i in range(self.num_examples):
            for j in range(X.shape[0]):
                kern[i,j] = Kpoly(self.X[i], X[j], self.deg)

        a = np.zeros(X.shape[0])
        for n in range(X.shape[0]):
            a[n] = np.sum(kern[:,n] * self.alpha * self.y) + self.b

        return np.sign(a)



def Kpoly(x, z, deg):
    """ Polynomial kernel function

    Arguments:
        x   - List of current values

        z   - List of taget values

        deg - Degree of polynomial to use

    Returns:
        Evaluated vale of kernel function
    """

    return (1 + np.dot(x, z))**deg



def Klin(x, z):
    """ Linear kernel function

    Arguments:
        x   - List of current values

        z   - List of taget values

    Returns:
        Evaluated vale of kernel function
    """

    return np.dot(x, z)