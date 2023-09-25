import numpy as np
import matplotlib.pyplot as plt
from Main import Run

values=["Dataset","Train",50] #or Kfold
# to plot graphs
def plot_graph(result_1, result_2, result_3, result_4):
    plt.figure(dpi=120)
    loc, result = [], []
    result.append(result_1)  # appending the result
    result.append(result_2)
    result.append(result_3)
    result.append(result_4)
    result = np.transpose(result)

    # labels for bars
    labels = ['RVM','BGRU','RF','NCV and stacked ensemble learning','Proposed R-CNN-TPOT']  # x-axis labels ############################
    tick_labels = ['MSE','RMSE','R-SQUARED','MAE']  #### metrics
    bar_width, s = 0.15, 0.025  # bar width, space between bars

    for i in range(len(result)):  # allocating location for bars
        if i == 0:  # initial location - 1st result
            tem = []
            for j in range(len(tick_labels)):
                tem.append(j + 1)
            loc.append(tem)
        else:  # location from 2nd result
            tem = []
            for j in range(len(loc[i - 1])):
                tem.append(loc[i - 1][j] + s + bar_width)
            loc.append(tem)

    # plotting a bar chart
    for i in range(len(result)):
        plt.bar(loc[i], result[i], label=labels[i], tick_label=tick_labels, width=bar_width)

    plt.legend(loc=(0.25, 0.25))# show a legend on the plot -- here legends are metrics
    plt.savefig("result.png")  # to show the plot



if values[1] == 'Train':
    tp = int(values[2]) / 100
else:
    tp = (int(values[2]) - 1) / int(values[2])  # k-fold calculation
dataset, tr_per = values[0], tp
dts = dataset
print("\n Reading data..")
print (dts,tr_per) #Dataset 0.9857142857142858
mse,rmse,r,mae = Run.callmain(dts,tr_per)
print (mse,rmse,r,mae)
plot_graph(mse,rmse,r,mae)
