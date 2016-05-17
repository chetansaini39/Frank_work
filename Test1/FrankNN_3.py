# This class if for second file BSN15_extended_training

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree  # import Decision tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics


class FrankNN_2:
    # file location
    fileLocation = "C:/Users/csaini/Documents/Frank Garcia/NN Data/csv/BSN15_extended_training.csv"
    signalData = pd.read_csv(fileLocation)  # read using Panda
    data = np.array(signalData)  # store whole data in NPARRAY
    print data.size  # size onf NP data array
    print 'rows, cols', data.shape  # row and cols
    print data.ndim  # number of axes /dimesnion
    print 'data type', data.dtype
    # print 'first row', data[0, 0:32]
    X = data[:, 0:32]  # all the rows and 0-31 cols
    Y = data[:, 32]  # store all Y
    x_test=X[0:1600] # store first 1600 rows into test
    y_test=Y[0:1600]

    print 'Size of Y : ', Y.size  # print the size of Y, just to check dimesnion are correct
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=.20, random_state=1)
    DT_Classifier = tree.DecisionTreeClassifier()
    DT_Classifier = DT_Classifier.fit(X_TRAIN, Y_TRAIN)  # train the model

    print 'prediction for first row: ', DT_Classifier.predict(data[0, 0:32])
    print 'plotting 1st line'
    plt.figure()
    plt.plot(X_TEST[0, 0:32], 'b-')  # take first line of test data and plot it
    plt.axhline(y=0, color='k')  # draws a horizontal line or axis line
    redDot = plt.plot(32, DT_Classifier.predict(X_TEST[0, 0:32]), 'ro',
                      label='Predicted spot')  # predict the output of first line on plot
    plt.legend([redDot], ['Predicted spot'])
    plt.show()

    plt.figure('Predicted values of first 32 samples')
    x = np.empty(32, dtype=int)  # create empty array of 32 size
    y = np.empty(32, dtype=float)
    for i in range(0, 32):
        x[i] = i
        y[i] = DT_Classifier.predict(X_TEST[i, 0:32])
    plt.subplot(211)
    plt.plot(x, y, 'r-')
    plt.subplot(212)
    plt.plot(Y_TEST[0:32], 'b-')  # plot the actual Y of first 32 test
    plt.show()

    # predict value of each label
    predicted = DT_Classifier.predict(X_TEST)  # predicted of all values
    print 'Predicted size', predicted.size
    print 'Target size', Y_TEST.size
    # print 'Accuracy', metrics.classification_report(y1, p1)
    print 'confusion matrix ', metrics.confusion_matrix(Y_TEST, predicted)
