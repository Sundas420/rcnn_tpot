from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

def classify(x,y,tr):
    tst = 1-tr
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=tr, test_size=tst)
    clf = DecisionTreeRegressor(max_depth=100)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    return pred