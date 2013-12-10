""" Module containing some scripts for plotting my results
"""

__author__ = 'Rob Argue'



import pickle as p
import matplotlib.pyplot as plt
import numpy as np



def plot_perceptron():
    """ Plots the results of the perceptrons
    """

    pts = p.load(open('classifiers/perceptrons_05.p', 'rb'))
    pt_x = []
    pt_y = []
    for pt in pts:
        pt_x.append(pt[0])
        pt_y.append(pt[2])

    apts = p.load(open('classifiers/averaged_perceptrons_05.p', 'rb'))
    apt_x = []
    apt_y = []
    for apt in apts:
        apt_x.append(apt[0])
        apt_y.append(apt[2])

    kpts = p.load(open('classifiers/kernelized_perceptrons_05_a.p', 'rb'))
    kpt_x = []
    kpt_y = []
    for kpt in kpts:
        kpt_x.append(kpt[0])
        kpt_y.append(kpt[2])

    plt.plot(pt_x,pt_y,apt_x,apt_y,kpt_x,kpt_y)
    plt.legend(('Perceptron', 'Averaged Perceptron', 'Kernelized Perceptron'),
        'bottom center')
    plt.xlabel('Max Iterations')
    plt.ylabel('Accuracy')
    plt.title('Fig. 1: Perceptrons')
    plt.show()

def plot_kernelized_perceptron():
    """ Plots the results of the kernelized perceptrons only
    """

    kpts = p.load(open('classifiers/kernelized_perceptrons_05_b.p', 'rb'))
    kpt_x = []
    kpt_y = []
    for kpt in kpts:
        kpt_x.append(kpt[0])
        kpt_y.append(kpt[2])

    plt.plot(kpt_x[:-1],kpt_y[:-1])
    plt.xlabel('Kernel Degree')
    plt.ylabel('Accuracy')
    plt.title('Fig. 2: Kernelized Perceptron')
    plt.show()