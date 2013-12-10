""" Module for loading in the NS2 data
"""

__author__ = 'Rob Argue'



import numpy as np
import csv
import pandas as pd



def CreateDataFrames(examples, features, outNum):
    """ Translates a CSV file into a pandas.DataFrame
    
    Arguments:
        examples    - Name of the CSV containing the data

        features    - Name of the file containing the column names for the CSV

        outNum      - Number to label the output files with

    Returns:
        (train, test, tune), each pandas.DataFrame's containing data sets for
        training, testing, and tuning broken up as 70 / 20 / 10 percent of the
        data.
    """

    values = np.genfromtxt(examples, delimiter=',')

    file = open(features, 'rb')
    reader = csv.reader(file)
    feats = reader.next()
    file.close()

    df =  pd.DataFrame(values,columns=feats)

    # Turn the features into categorical ones
    Categoricalize(df, feats)

    # Turn labels into +/- 1
    df['Winner'] = np.sign(df['Winner'] - 1.5)

    # Following code adapted from PA01
    nsamples = df.shape[0]
    ntest = np.floor(.2 * nsamples)
    ntune = np.floor(.1 * nsamples)

    # we want to make this reporducible so we seed the random number generator
    np.random.seed(1)
    all_indices = np.arange(nsamples)+1
    np.random.shuffle(all_indices)
    test_indices = all_indices[:ntest]
    tune_indices = all_indices[ntest:(ntest+ntune)]
    train_indices = all_indices[(ntest+ntune):]

    train = df.ix[train_indices,:]
    tune = df.ix[tune_indices,:]
    test = df.ix[test_indices,:]

    pd.save(train, 'data/train/train' + outNum + '.pdat')
    pd.save(tune, 'data/train/tune' + outNum + '.pdat')
    pd.save(test, 'data/test/test' + outNum + '.pdat')

    return train, tune, test



def Categoricalize(df, feats):
    """ Turns the continuous raw data into categorical data. Specific to the
    current setup of features for the NS2 data 

    Arguments:
        df      - pandas.DataFrame contatining the data to categoricalize

        feats   - List of features in the DataFrame
    """

    # I chose this, no idea how good it is
    time_cuts = np.array([0,1,2,3,4,5,7.5,10,12.5,15,20,25,30,40,50,60,90,120])
    #time_cuts = np.array([0,1,2,5,10,15,30,60,120])
    time_cuts *= 60

    # THIS IS SPECIFIC TO THE CURRENT DATA SETUP
    time_feats = feats[:1] + feats[5:-1]

    for feat in time_feats:
        df[feat]=pd.cut(df[feat],time_cuts)

    tvt_feats = feats[1:5]
    tvt_cuts = np.array([0, 0.1, 0.2, 0.25, 0.333, 0.5, 0.667, 0.75, 1, 1.333, 1.5, 2, 3, 4, 5, 10, 10000])
    #tvt_cuts = np.array([0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 10000])

    for feat in tvt_feats:
        df[feat]=pd.cut(df[feat],tvt_cuts)
