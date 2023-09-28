import os

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, LearningCurveDisplay, RandomizedSearchCV
from sklearn.inspection import permutation_importance
import math
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer, precision_score, \
    f1_score
from matplotlib import pyplot
import _pickle as cPickle
from categoriesList import cat
from landuseList import land

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def complex_operation(input):
    categoriesMatrix = np.array([])
    landuseMatrix = np.array([])
    landuseCats = list(map(str, [111,112,121,122,123,124,131,132,133,141,142,211,212,213,221,222,223,231,241,242,243,244,311,312,313,321,322,323,324,331,332,333,334,335,411,412,421,422,423,511,512,521,522,523]))
    cocoClasses = list(map(str, range(1, 81)))
    index = 0
    for i, r in input.iterrows():
        landuse = r["Landuse"].split(",")
        categories = r["categories"].split(",")
        mask = np.isin(cat, categories)
        maskLanduse = np.isin(landuseCats, landuse)
        if index == 0:
            categoriesMatrix = [mask.astype(int)]
            landuseMatrix = [maskLanduse.astype(int)]

        else:
            categoriesMatrix = np.append(categoriesMatrix, [mask.astype(int)], axis=0)
            landuseMatrix = np.append(landuseMatrix, [maskLanduse.astype(int)], axis=0)
        index += 1

    return categoriesMatrix, landuseMatrix


def run_complex_operations(operation, pool):
    df = pd.read_csv('/Users/quintengroenveld/PycharmProjects/flickrHandler/swiss_data_v1.csv', sep=';', keep_default_na=False)
    #df = df[:100]
    input_data1 = df.iloc[:df.shape[0] // 10]
    input_data10 = df.iloc[(df.shape[0] // 10):(df.shape[0] // 10) * 2]
    input_data2 = df.iloc[(df.shape[0] // 10) * 2:(df.shape[0] // 10) * 3]
    input_data3 = df.iloc[(df.shape[0] // 10) * 3:(df.shape[0] // 10) * 4]
    input_data4 = df.iloc[(df.shape[0] // 10) * 4:(df.shape[0] // 10) * 5]
    input_data5 = df.iloc[(df.shape[0] // 10) * 5:(df.shape[0] // 10) * 6]
    input_data6 = df.iloc[(df.shape[0] // 10) * 6:(df.shape[0] // 10) * 7]
    input_data7 = df.iloc[(df.shape[0] // 10) * 7:(df.shape[0] // 10) * 8]
    input_data8 = df.iloc[(df.shape[0] // 10) * 8:(df.shape[0] // 10) * 9]
    input_data9 = df.iloc[(df.shape[0] // 10) * 9:]
    input = [input_data1, input_data10, input_data2, input_data3, input_data4, input_data5, input_data6, input_data7,
             input_data8, input_data9]
    x = pool.map(operation, input)
    df_new = pd.concat([pd.DataFrame(x[0][0]), pd.DataFrame(x[1][0]), pd.DataFrame(x[2][0]), pd.DataFrame(x[3][0]),
                        pd.DataFrame(x[4][0]), pd.DataFrame(x[5][0]), pd.DataFrame(x[6][0]), pd.DataFrame(x[7][0]),
                        pd.DataFrame(x[8][0]), pd.DataFrame(x[9][0])], axis=0)
    df_new1 = pd.concat([pd.DataFrame(x[0][1]), pd.DataFrame(x[1][1]), pd.DataFrame(x[2][1]), pd.DataFrame(x[3][1]),
                         pd.DataFrame(x[4][1]), pd.DataFrame(x[5][1]), pd.DataFrame(x[6][1]), pd.DataFrame(x[7][1]),
                         pd.DataFrame(x[8][1]), pd.DataFrame(x[9][1])], axis=0)
    df_new = df_new.reset_index()
    df_new1 = df_new1.reset_index()
    df = df.reset_index()
    df_new = pd.concat([df[['latitude', 'longitude', 'user_guid', 'categories', 'post_create_date', 'Noise', 'Elevation']], df_new, df_new1], axis=1)
    df_new.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis2/Swiss_finaldata_for_RF_v1.csv', sep=";", index=False)
    return df_new

def predict_Random_forest(x):
    Xdata = x.iloc[:, np.r_[5:7, 8:373, 373:417]]
    path = '/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/model_n600'
    if os.path.isfile(path):
        print("running improved model")
        with open('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/model_n600', 'rb') as f:
            regressor = cPickle.load(f)
    y_pred = regressor.predict(Xdata)
    print(y_pred)
    exit()
    return y_pred

if __name__ == '__main__':
    # processes_count=10
    # processes_pool = Pool(processes_count)
    # data = run_complex_operations(complex_operation, processes_pool)
    data = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis2/Swiss_finaldata_for_RF_v1_1.csv', sep=';',
                       keep_default_na=False)
    predict_Random_forest(data)
    # dict = {}
    # indexCat = 0
    # indexLand = 0
    # del data['index.1']
    # for i in range(0, len(data.columns)):
    #     if i < 8:
    #         dict[data.columns[i]] = data.columns[i]
    #         continue
    #     elif i >= 8 and i < 373:
    #         dict[data.columns[i]] = cat[indexCat]
    #         indexCat+=1
    #     elif i >= 373 and i < 418:
    #         dict[data.columns[i]] = land[indexLand]
    #         indexLand+=1
    # data1 = data.rename(columns=dict)
    # data1.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis2/Swiss_finaldata_for_RF_v1_1.csv', sep=";", index=False)
    # # predict_Random_forest(data1)

