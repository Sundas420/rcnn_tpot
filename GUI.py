import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from Main import Run
sg.change_look_and_feel('DarkBrown5')    # look and feel theme


# Designing layout
layout = [[sg.Text("\t\t\tSelect_dataset  "), sg.Combo(['Dataset'],size=(15, 20)),sg.Text("\n")],
          [sg.Text("\t\t\tSelect\t         "), sg.Combo(['Trainingset(%)', 'K-value'], size=(15, 20)), sg.InputText(size=(10, 23)),sg.Text("\t\t"),sg.Button("START", size=(10, 2)), sg.Text("\n")],
          [sg.Text("\n"), sg.Text("\t\t      RVM\t\t\t      BGRU\t\t\tRF\tNCV and stacked ensemble learning        Proposed R-CNN-TPOT ")],
          [sg.Text('MSE\t     '), sg.In(key='11',size=(23,20)), sg.In(key='12',size=(23,20)), sg.In(key='13',size=(23,20)), sg.In(key='14',size=(23,20)),sg.In(key='15',size=(23,20)),sg.Text("\n")],
          [sg.Text('RMSE\t     '), sg.In(key='21',size=(23,20)), sg.In(key='22',size=(23,20)), sg.In(key='23',size=(23,20)), sg.In(key='24',size=(23,20)),sg.In(key='25',size=(23,20)),sg.Text("\n")],
          [sg.Text('R-SQUARED'), sg.In(key='31', size=(23, 20)), sg.In(key='32', size=(23, 20)),sg.In(key='33', size=(23, 20)), sg.In(key='34', size=(23, 20)),sg.In(key='35', size=(23, 20)), sg.Text("\n")],
          [sg.Text('MAE\t     '), sg.In(key='41', size=(23, 20)), sg.In(key='42', size=(23, 20)),sg.In(key='43', size=(23, 20)), sg.In(key='44', size=(23, 20)),sg.In(key='45', size=(23, 20)), sg.Text("\n")],
          [sg.Text('\t\t\t\t\t\t\t           '), sg.Button('Run Graph'), sg.Button('CLOSE')]]


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
    plt.show()  # to show the plot


# Create the Window layout
window = sg.Window('116604', layout)

# event loop
while True:
    event, values = window.read()  # displays the window
    if event == "START":
        if values[1] == 'TrainingData(%)':
            tp = int(values[2]) / 100
        else:
            tp = (int(values[2]) - 1) / int(values[2])  # k-fold calculation
        dataset, tr_per = values[0], tp
        dts = dataset
        print("\n Reading data..")
        print (dts,tr_per) #Dataset 0.9857142857142858
        mse,rmse,r,mae = Run.callmain(dts,tr_per)


        window.element('11').Update(mse[0])
        window.element('12').Update(mse[1])
        window.element('13').Update(mse[2])
        window.element('14').Update(mse[3])
        window.element('15').Update(mse[4])

        window.element('21').Update(rmse[0])
        window.element('22').Update(rmse[1])
        window.element('23').Update(rmse[2])
        window.element('24').Update(rmse[3])
        window.element('25').Update(rmse[4])

        window.element('31').Update(r[0])
        window.element('32').Update(r[1])
        window.element('33').Update(r[2])
        window.element('34').Update(r[3])
        window.element('35').Update(r[4])

        window.element('41').Update(mae[0])
        window.element('42').Update(mae[1])
        window.element('43').Update(mae[2])
        window.element('44').Update(mae[3])
        window.element('45').Update(mae[4])

    if event == 'Run Graph':
        plot_graph(mse,rmse,r,mae)
    if event == 'CLOSE':
        window.close()
        break
