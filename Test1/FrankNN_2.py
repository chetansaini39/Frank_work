import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree  # import Decision tree


class FrankNN_2:
    # file location
    fileLocation = "C:/Users/csaini/Documents/Frank Garcia/NN Data/csv/testfile2.csv"
    signalData = pd.read_csv(fileLocation)  # read using Panda
    data = np.array(signalData)  # store whole data in NPARRAY
    print data.size  # size onf NP data array
    print 'rows, cols', data.shape  # row and cols
    print data.ndim  # number of axes /dimesnion
    print 'data type', data.dtype
    print 'first row', data[0, 0:32]
    X = data[:, 0:32]  # all the rows and 0-31 cols
    Y = data[:, 32]  # store all Y
    print 'Size of Y', Y.size       # print the size of Y, just to check dimesnion are correct
    DT_Classifier = tree.DecisionTreeClassifier()
    DT_Classifier = DT_Classifier.fit(X, Y)
    print 'prediction for first row: ', DT_Classifier.predict(data[2,0:32])
    print 'plotting 1st line'
    plt.figure()
    plt.plot(data[0, 0:32], 'b-')
    plt.axhline(y=0, color='k')# draws a horizontal line or axis line
    plt.show()