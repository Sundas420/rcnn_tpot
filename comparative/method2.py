import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

from Main.metrics import metric


def classify(x,y,tr,MSE, RMSE, R_sq, MAE):
    tst = 1-tr
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=tr, test_size=tst)
    clf = DecisionTreeRegressor(max_depth=100)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    mse, rmse, r, mae = metric(y_test,pred)

    MSE.append(mse)
    RMSE.append(rmse)
    R_sq.append(r)
    MAE.append(mae)