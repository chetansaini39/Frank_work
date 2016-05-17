import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_validation import train_test_split

class FrankNN:
    url = "C:/Users/csaini/Documents/Frank Garcia/NN Data/csv/testfile2.csv"
    signalData = pd.read_csv(url)
    rng = np.random.RandomState(1)
    # print signalData[signalData.columns[2]]
    # print np.corrcoef(signalData[signalData.columns[0]], signalData[signalData.columns[1]])
    # print(signalData)
    # print np.corrcoef(signalData,signalData)
    X = np.array(signalData.ix[:, 0:32],dtype=float)
    y = np.array(signalData.ix[:, 32],dtype=float)
    print 'y',y.size
    count = 0
    x_axis = np.empty(32, dtype=int)
    for i in X[0, 0:32]:
        x_axis[count]=count
        count += 1
    # print X[0, 0:32]
    print 'Length of x, x_axis', len(X[0, 0:32]), len(x_axis)
    for i in X[i, 0:32]:
            plt.scatter(x_axis,  X[i, 0:32], c="g",linewidths=1)
            print 'Scatter Plotting data of row ', i
    plt.show()
    # plt.figure()
    # plt.plot(x_axis, X[1, 0:32], c="r")
    # plt.show()


    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=4)

    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                               n_estimators=300, random_state=rng)
    print X
    print "X dim", X.size
    print "y dim", y.size
    print "x_axis dim", x_axis.size

    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)
    #
    # # X1 = np.linspace(0, 6, 100)[:, np.newaxis]
    # # y1 = np.sin(X1).ravel() + np.sin(6 * X1).ravel() + rng.normal(0, 0.1, X1.shape[0])
    # #
    # # print X1, y1
    # # Plot the results
    plt.figure()
    plt.scatter(x_axis,  X[i, 0:32],c="k", label="training sample")
    plt.plot(x_axis, y_1, c="g", linewidth=2)
    plt.plot(x_axis, y_2, c="r", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show()
