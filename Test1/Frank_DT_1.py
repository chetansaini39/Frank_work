# This class if for second file BSN15_extended_training

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree  # import Decision tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics


class Frank_DT_1:
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
    x_train = X[0:1700]  # store first 1600 rows into test
    y_train = Y[0:1700]
    x_test = X[1700:1800]  # This is test sample
    y_test = Y[1700:1800]

    DT_Classifier = tree.DecisionTreeClassifier(max_depth=15,criterion='gini')
    DT_Classifier = DT_Classifier.fit(x_train, y_train)  # train the model
    print 'plotting 1st line', x_test[0, 0:32]
    plt.figure()
    plt.plot(x_test[0, 0:32], 'b-')
    plt.axhline(y=0, color='k')  # draws a horizontal line or axis line
    redDot = plt.plot(32, DT_Classifier.predict(x_test[0, 0:32]), 'ro',
                      label='Predicted spot')  # predict the output of first line on plot
    plt.show()

    plt.figure('Predicted values of first 32 samples')
    x = np.empty(64, dtype=int)  # create empty array of 32 size
    y = np.empty(64, dtype=float)
    for i in range(0, 64):
        x[i] = i
        y[i] = DT_Classifier.predict(x_test[i, 0:32])
    plt.subplot(211)
    plt.plot(x, y, 'r-')
    plt.subplot(212)
    plt.plot(y_test[0:64], 'b-')  # plot the actual Y of first 32 test
    plt.show()

    Y_TEST = np.array(["%.2f" % w for w in y_test])
    y_predicted = np.array(["%.2f" % w for w in DT_Classifier.predict(x_test)])
    print 'Accuracy Score ', metrics.accuracy_score(Y_TEST, y_predicted)
    print 'Confusion MAtrix', metrics.confusion_matrix(Y_TEST, y_predicted)
