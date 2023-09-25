from Main import feature_selection
from performance import decision_tree,RCNN,CNN,CNN2,CNN3,CNN4
# from Main import process

def callmain(dts,tr):
    # process.preprocessing()                           #preprocessing
    data, target = feature_selection.feat()             #feature selection
    MSE, RMSE, R_sq, MAE = [], [],[],[]

    #-----------------------------Proposed--------------------------------

    O1 = decision_tree.classify(data,target,tr)
    O2 = RCNN.classify(data,target,tr,O1)
    CNN.cnn(O2,target,tr,MSE, RMSE, R_sq, MAE,10)
    CNN2.cnn(O2,target,tr,MSE, RMSE, R_sq, MAE,20)
    CNN3.cnn(O2,target,tr,MSE, RMSE, R_sq, MAE,30)
    CNN4.cnn(O2,target,tr,MSE, RMSE, R_sq, MAE,40)
    return MSE, RMSE, R_sq, MAE

