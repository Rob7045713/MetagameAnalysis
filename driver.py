""" Main driver for testing NS2 data
"""

__author__ = 'Rob Argue'



import dectree.DTree as dt
import pickle as p
import Loader as l
import analysis as an
import Perceptron as pt
import AveragedPerceptron as apt
import KernelizedPerceptron as kpt



def LoadAndTreeEval():
    """ Load the data and create a decision tree
    
    Returns:
        DTree generated for the data
    """

    train, test, tune = l.CreateDataFrames('data/Build229Data.txt', 'data/FeatureNames.txt')
    tree = dt.get_tree(train, 'Winner', 10)
    p.dump(tree, open('classifiers/dtree03.p','wb'))
    pred = tree.predict(test)
    true = test['Winner'].values
    print an.accuracy(pred, true)
    print an.f1_score(pred, true)

    return tree



def MakePerceptrons(train, test, label):
    """ Create and evaluate different types of perceptrons

    Arguments:
        train   - pandas.DataFrame containing the training data

        test    - pandas.DataFrame containing the testing data

        label   - Name of the label column in the DataFrame's
    
    Returns:
        (List of Perceptron, List of AveragedPerceptron,
         List of KernelizedPerceptron) generated for the data
    """

    true = test[label].values
    pts = []
    apts = []
    kpts = []

    for i in [1,5,10,25,50]:
        print "i = " + str(i)
        print "PT working"
        ptron = pt.Perceptron(train, i)
        pred = ptron.predict(test, label)
        acc = an.accuracy(pred, true)
        pts.append((i, ptron, acc))
        print "PT " + str(acc)

        print "APT working"
        ap = apt.AveragedPerceptron(train, i)
        pred = ap.predict(test, label)
        acc = an.accuracy(pred, true)
        apts.append((i, ap, acc))
        print "APT " + str(acc)

        print "KPT working"
        kp = kpt.KernelizedPerceptron(train, label, i, 2)
        pred = kp.predict(test, label)
        acc = an.accuracy(pred, true)
        kpts.append((i, kp, acc))
        print "KPT " + str(acc)

    p.dump(pts, open('classifiers/perceptrons.p','wb'))
    p.dump(apts, open('classifiers/averaged_perceptrons.p','wb'))
    p.dump(kpts, open('classifiers/kernelized_perceptrons.p','wb'))

    return pts, apts, kpts



def MakeKernelizedPerceptrons(train, test, label):
     """ Create and evaluate kernelized perceptrons

    Arguments:
        train   - pandas.DataFrame containing the training data

        test    - pandas.DataFrame containing the testing data

        label   - Name of the label column in the DataFrame's
    
    Returns:
        (List of KernelizedPerceptron) generated for the data
    """

    true = test[label].values
    kpts = []

    for i in [2,3,5,10,25,50]:

        print "KPT working"
        kp = kpt.KernelizedPerceptron(train, label, 25, i)
        pred = kp.predict(test, label)
        acc = an.accuracy(pred, true)
        kpts.append((i, kp, acc))
        print "KPT " + str(acc)

    p.dump(kpts, open('classifiers/kernelized_perceptrons01.p','wb'))

    return kpts