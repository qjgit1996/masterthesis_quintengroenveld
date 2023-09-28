import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay


def analyze(data):
    df = data.sample(n=200, random_state=0)
    df.to_csv('/Users/quintengroenveld/PycharmProjects/discussion/200random_samples.csv', sep=";")

def compare(data):
    def absolute_value(val):
        a = np.round(val / 100. * sizes.sum(), 0)
        return int(a)
    df = data.groupby('OwnPrediction').count().iloc[: , :1]
    labels = 'Useable Scenic Prediction','Indoor Images'
    sizes = np.array([(df.iloc[0][0]+ df.iloc[1][0]+ df.iloc[2][0]+ df.iloc[3][0]), df.iloc[4][0]])
    fig = plt.figure(figsize=(10, 5))
    # creating the bar plot
    plt.bar(labels, sizes, color='maroon',
            width=0.4)
    plt.ylabel("Count")
    plt.show()

def compare2(data):
    data = data[data['OwnPrediction'].str.contains(r'^\d+$')]
    #data['OwnPrediction'] = data['OwnPrediction'].astype(int)
    #data = data[data['OwnPrediction']>5]
    y_pred_test = data['Predictions'].values.ravel()
    print(y_pred_test)
    minValues = [1, 3.25, 5.5, 7.75]
    maxValues = [3.25, 5.5, 7.75, 10.0]
    index = 0
    minDifferences = []
    for i in y_pred_test:
        minimum = minValues[int(i)-1]
        maximum = maxValues[int(i)-1]
        op = float(data['OwnPrediction'].values.ravel()[index])
        diffMin = op - minimum
        diffMax = op - maximum
        if diffMax<0 and diffMin>0:
            minDifferences.append(0)
        if diffMax<0 and diffMin<0:
            minDifferences.append(min([abs(diffMin), abs(diffMax)]))
        if diffMax>0 and diffMin>0:
            minDifferences.append(min([abs(diffMin), abs(diffMax)]))
        index+=1

    fig, ax = plt.subplots(figsize=(10, 30))
    ax.hist(minDifferences, edgecolor='black', bins=[0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5], align='left')
    ax.set_xticks([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5])
    ax.set_xticklabels((0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5))
    plt.xlabel("Difference")
    plt.ylabel("Count")
    plt.title("Difference between Class Value and Vote Average")
    # Show plot
    plt.show()

def cmPlot(data):
    data = data[data['OwnPrediction'].str.contains(r'^\d+$')]
    data['OwnPrediction'] = data['OwnPrediction'].astype(int)
    y_pred_test = data['OwnPrediction'].values.ravel()
    y_test = data['Predictions'].values.ravel()
    cm = confusion_matrix(y_test, y_pred_test)
    print("F1 score: " + str(f1_score(y_test, y_pred_test, average=None)))

    Cm = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["0.999-3.25", "3.25-5.5", "5.5-7.75", "7.75-10.0"]).plot()
    plt.ylabel("Classifier Model Prediction")
    plt.xlabel("Personal Manual Classification")
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # data = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_classifier.csv',sep=';',keep_default_na=False)
    # df = data.loc[(data['Color1'] > 0) & (data['Color2'] > 0) & (data['Color3'] > 0)]
    # df.to_csv('/Users/quintengroenveld/PycharmProjects/discussion/swiss_results_classifier_without_not_found.csv', sep=";")
    # analyze(data)
    data = pd.read_csv('/Users/quintengroenveld/PycharmProjects/discussion/200random_samples.csv', sep=';',
                    keep_default_na=False)
    compare(data)
    #cmPlot(data)
