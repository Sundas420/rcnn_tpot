from Main import feature_selection
from comparative import method1
from comparative import method2
from comparative import method3
from comparative import method4
from Proposed import decision_tree,RCNN,CNN
# from Main import process

def callmain(dts,tr):

    # process.preprocessing()                           #preprocessing
    data, target = feature_selection.feat()             #feature selection
    MSE, RMSE, R_sq, MAE = [], [],[],[]

    #-----------------------------Proposed--------------------------------

    O1 = decision_tree.classify(data,target,tr)
    O2 = RCNN.classify(data,target,tr,O1)
    CNN.cnn(O2,target,tr,MSE, RMSE, R_sq, MAE)

    #---------------------------comparative-------------------------------

    method4.classify(data,target,tr,MSE, RMSE, R_sq, MAE)
    method3.classify(data,target,tr,MSE, RMSE, R_sq, MAE)
    method2.classify(data,target,tr,MSE, RMSE, R_sq, MAE)
    method1.classify(data,target,tr,MSE, RMSE, R_sq, MAE)

    return MSE, RMSE, R_sq, MAE

