import numpy as np
import pandas as pd
from scipy.spatial import distance

def feat():
    X = pd.read_csv(r'D:\rcnn_tpot\Main\Feature.csv',header=None)
    Y = pd.read_csv(r'D:\rcnn_tpot\Main\Label.csv',header=None)
    x = np.array(X)
    y = np.array(Y)
    print (len(x[0]))
    y = np.nan_to_num(y)

    #---------------------------Canberra distance-----------------------------

    cd = []
    for i in range(X.shape[1]):
        d=distance.canberra(x[:,i], y[:,0])
        cd.append(d)
    sel = int((np.sum(cd))/max(cd))

    feat_sel = x[:,:sel]
    return feat_sel,y

