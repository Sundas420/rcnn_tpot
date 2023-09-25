import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from Main.metrics import metric


def classify(X,Y,tr,MSE, RMSE, R_sq, MAE):

    tst = 1-tr
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=tr, test_size=tst)
    clf = KNeighborsRegressor(n_neighbors=5, p=2, weights="distance")
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    mse, rmse, r, mae = metric(y_test,pred)

    MSE.append(mse)
    RMSE.append(rmse)
    R_sq.append(r)
    MAE.append(mae)

    for i in range(0, len(MSE)):
        for j in range(i + 1, len(MSE)):
            if (MSE[i] < MSE[j]):
                temp = MSE[i]
                MSE[i] = MSE[j]
                MSE[j] = temp

    for i in range(0, len(RMSE)):
        for j in range(i + 1, len(RMSE)):
            if (RMSE[i] < RMSE[j]):
                temp = RMSE[i]
                RMSE[i] = RMSE[j]
                RMSE[j] = temp

    for i in range(0, len(R_sq)):
        for j in range(i + 1, len(R_sq)):
            if (R_sq[i] < R_sq[j]):
                temp = R_sq[i]
                R_sq[i] = R_sq[j]
                R_sq[j] = temp

    for i in range(0, len(MAE)):
        for j in range(i + 1, len(MAE)):
            if (MAE[i] < MAE[j]):
                temp = MAE[i]
                MAE[i] = MAE[j]
                MAE[j] = temp