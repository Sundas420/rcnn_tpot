import random

import numpy as np
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

def classify(xx,yy,tr,o1):
    o2 = []
    tst = 1-tr
    alpha =5
    X_train, X_test, y_train, y_test = train_test_split(xx, yy,
                                                        train_size=tr, test_size=tst)
    model = TPOTRegressor(generations=5, population_size=50, scoring='neg_mean_absolute_error', cv=4, verbosity=2,
                          random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)
    w = random.randint(1,5)
    for i in range(xx.shape[1]):
        x = xx[:, i]
        o1 = np.resize(o1, (len(x),))
        z = alpha * x * w + 1 / 2 * o1
        o2.append(z)
    o2 = np.transpose(o2)
    return o2