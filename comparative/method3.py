from Main.metrics import metric
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def classify(X,Y,tr,MSE, RMSE, R_sq, MAE):

    tst = 1-tr
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=tr, test_size=tst)
    clf = GradientBoostingRegressor(loss="squared_error",criterion="friedman_mse")
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    mse, rmse, r, mae = metric(y_test,pred)

    MSE.append(mse)
    RMSE.append(rmse)
    R_sq.append(r)
    MAE.append(mae)