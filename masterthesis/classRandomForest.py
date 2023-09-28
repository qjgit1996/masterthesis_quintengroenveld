import os
from datetime import datetime

import datetime as datetime
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, LearningCurveDisplay, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

from categoriesList import cat

from landuseList import land
import math
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer, precision_score, \
    f1_score
from matplotlib import pyplot
import _pickle as cPickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def random_forest_balanced(X_train_id, y_train, X_test, y_test):
    X_train = X_train_id
    cls = RandomForestClassifier(n_estimators= 2000, min_samples_split=10, min_samples_leaf= 5, max_leaf_nodes=200, max_depth=10, class_weight=None,
                                      random_state=0,
                                      warm_start=True)
    cls.fit(X_train, y_train)
    y_pred_test = cls.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    print("F1 score: " + str(f1_score(y_test, y_pred_test, average=None)))

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0.999-3.25", "3.25-5.5", "5.5-7.75", "7.75-10.0"]).plot()
    pyplot.show()

    with open('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/cls_model_final_v1', 'wb') as f:
        cPickle.dump(cls, f)
    return y_pred_test, y_test

def random_forest(data):
    goodData = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/featureImportance_class_min.csv', sep=';',
                        keep_default_na=False)
    goodData = goodData.loc[goodData.iloc[:,1] > 0.0]
    y = data.iloc[:, 502].values.ravel()
    for i in data.columns:
        print(i)
        if i not in goodData.iloc[:,0].values.ravel():
            print("dropped")
            data = data.drop(i, axis=1)
    data.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/deletedFeat_2.csv', sep=";")
    x = data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    cls = RandomForestClassifier(n_estimators= 1000, min_samples_split=20, min_samples_leaf= 100, max_leaf_nodes=200, max_depth=8,
                                      random_state=0,
                                      warm_start=True)
    cls.fit(X_train, y_train)
    y_pred_test = cls.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    print("F1 score: " + str(f1_score(y_test, y_pred_test, average=None)))

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0.999-3.25", "3.25-5.5", "5.5-7.75", "7.75-10.0"]).plot()
    pyplot.show()

    with open('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/cls_model_n' + '1000_eq4classes_min_min', 'wb') as f:
        cPickle.dump(cls, f)
    return y_pred_test, y_test

def searchParamBalanced(X_train, y_train, X_test, y_test):
    param_dist = {'n_estimators': np.array([100, 200, 500, 1000, 1500, 2000, 5000]),
                  'max_depth': np.array([1, 3,  5, 8, 10, 20]),
                  'min_samples_split': np.array([3, 5, 10, 20, 50, 100]),
                  'max_leaf_nodes': np.array([3, 5, 10, 20, 100, 200]),
                  'min_samples_leaf': np.array([5, 10, 20, 50, 100, 200]),
                  'max_features': np.array([10, 50, 100, 200, 300, 600]),
                  'class_weight': [
                      'balanced',
                      None
                    ]
                  }

    # Create a random forest classifier
    rf = RandomForestClassifier(random_state=0)
    scorer = make_scorer(f1_score, average='micro')
    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf,
                                     param_distributions=param_dist,
                                     n_iter=10,
                                     cv=5,
                                     scoring=scorer,
                                     random_state=1)

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)
    best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:', rand_search.best_params_)

def searchParam(data):
    x = data.iloc[:, np.r_[7:12, 13:502]]
    y = data.iloc[:, 502].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    param_dist = {'n_estimators': np.array([100, 200, 500, 1000, 1500, 2000]),
                  'max_depth': np.array([1, 3,  5, 8, 10, 20]),
                  'min_samples_split': np.array([3, 5, 10, 20, 50, 100]),
                  'max_leaf_nodes': np.array([3, 5, 10, 20, 100, 200]),
                  'min_samples_leaf': np.array([5, 10, 20, 50, 100, 200]),
                  'max_features': np.array([10, 50, 100, 200, 300, 415]),
                  'class_weight': [
                      'balanced',
                      None
                    ]
                  }

    # Create a random forest classifier
    rf = RandomForestClassifier(random_state=0)
    scorer = make_scorer(f1_score, average='micro')
    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf,
                                     param_distributions=param_dist,
                                     n_iter=10,
                                     cv=5,
                                     scoring=scorer,
                                     random_state=1)

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)
    best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:', rand_search.best_params_)

def classDist(data):
    # Creating histogram
    data2 = data['SimpleValue'].value_counts()
    df = pd.Series(data2).sort_index()
    plt.bar(range(len(df)), df.values, align='center')
    plt.xticks(range(len(df)), df.index.values, size='small')
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title('Class Count')
    plt.show()
    # fig, ax = plt.subplots()
    # ax.hist(data["SimpleValue"], align="mid")
    # # Show plot
    # plt.show()

def valueDist(data):
    x = np.array(data['Average'])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(x, bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.xlabel("Score")
    plt.ylabel("Count")
    # Show plot
    plt.show()
    bins = pd.qcut(np.array(data['Average']), 5)
    print(bins)


def balancedTraining(data):
    x = data.iloc[:, np.r_[7:502]]
    y = data.iloc[:, np.r_[502]]
    # x = data.iloc[:, np.r_[0:47]]
    # y = data.iloc[:, 47].values.ravel()
    # Assuming X is your feature data and y is your target variable

    # Split the data stratified by y
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    # Combine the training data and labels
    combined_data = np.column_stack((X_train, y_train))

    # Separate data by class labels
    class_data = {}
    for label in np.unique(y):
        class_data[label] = combined_data[combined_data[:, -1] == label]

    # Calculate the minimum number of samples per class
    min_samples = min(len(class_data[label]) for label in np.unique(y))

    # Create balanced subsets
    balanced_train_data = np.vstack([class_data[label][:min_samples] for label in np.unique(y)])
    X_train_balanced = balanced_train_data[:, :-1]
    y_train_balanced = balanced_train_data[:, -1]

    # Shuffle the balanced training data
    shuffle_indices = np.random.permutation(len(X_train_balanced))
    X_train_balanced = X_train_balanced[shuffle_indices]
    y_train_balanced = y_train_balanced[shuffle_indices]
    # fig, ax = plt.subplots()
    # ax.hist(y_train_balanced, align="mid")
    # # Show plot
    # plt.show()

    return X_train_balanced, y_train_balanced, X_test, y_test

def retrain_Random_forest(x_train, y_train, x_test, y_test, index, n_estimators):
    path = '/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/cls_model_final_v1'
    if os.path.isfile(path):
        print("running improved model")
        with open(path, 'rb') as f:
            cls = cPickle.load(f)
            cls.set_params(n_estimators= n_estimators, min_samples_split=10, min_samples_leaf= 5, max_leaf_nodes=200, max_depth=10, class_weight=None,
                                      random_state=0,
                                      warm_start=True)
    else:
        cls = RandomForestClassifier(n_estimators=1000, min_samples_split=10, min_samples_leaf= 50, max_leaf_nodes=100,max_features= 200, max_depth=20,
                                          random_state=0,warm_start=True)
    cls.fit(x_train, y_train)
    y_pred_test = cls.predict(x_test)
    cm = confusion_matrix(y_test, y_pred_test)
    print("F1 score: " + str(f1_score(y_test, y_pred_test, average=None)))

    Cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0.999-3.25", "3.25-5.5", "5.5-7.75", "7.75-10.0"]).plot()
    Cm.figure_.savefig('cm_plots/confusion_matrix_' + str(index) +'.png')
    pyplot.close()
    with open('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/cls_model_final_v'+str(index), 'wb') as f:
        cPickle.dump(cls, f)
    return y_pred_test, y_test


def showFeautreImportance(data):
    data_X = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/deletedFeat.csv', sep=';',
                         keep_default_na=False)
    x = data_X
    x.columns = x.columns.astype(str)
    y = data.iloc[:, 502].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    with open('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/cls_model_n1000_eq4classes_extrafeatures', 'rb') as f:
        regressor = cPickle.load(f)
    result = permutation_importance(
        regressor, X_test, y_test, n_repeats=10, random_state=1, n_jobs=2
    )

    forest_importances = pd.Series(result.importances_mean, index=x.columns)
    forest_importances.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/featureImportance_class_min.csv', sep=";")

def predict_Random_forest(x):
    Xdata = x.iloc[:, np.r_[5:499]]
    path = '/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/cls_model_final_v1'
    print("running improved model")
    with open(path, 'rb') as f:
        cls = cPickle.load(f)
    y_pred = cls.predict(Xdata)
    x['Predictions'] = y_pred.tolist()
    return x

def simplifyData(data):
    data['Color1'] = data['Color1'].astype(str)
    data['Color2'] = data['Color2'].astype(str)
    data['Color3'] = data['Color3'].astype(str)
    data[['Color1', 'Color2', 'Color3']] = data[['Color1', 'Color2', 'Color3']].apply(LabelEncoder().fit_transform)
    data.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_validation_final_v3.csv', sep=";")

def ColorEncoder(data):
    df = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_finaldata_for_RF_v2_1.csv',
                       sep=';',
                       keep_default_na=False)
    df2 = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_finaldata_for_RF_v2_1_1.csv',
                     sep=';',
                     keep_default_na=False)
    for i, r in data.iterrows():
        ind1 = df.index[df.Color1 == r['Color1']]
        ind2 = df.index[df.Color2 == r['Color2']]
        ind3 = df.index[df.Color3 == r['Color3']]
        color1 = df2['Color1'][ind1]
        color2 = df2['Color2'][ind2]
        color3 = df2['Color3'][ind3]
        if ind1.empty:
            ind1 = df.index[df.Color2 == r['Color1']]
            if ind1.empty:
                ind1 = df.index[df.Color3 == r['Color1']]
        if ind2.empty:
            ind2 = df.index[df.Color1 == r['Color2']]
            if ind2.empty:
                ind2 = df.index[df.Color3 == r['Color2']]
        if ind3.empty:
            ind3 = df.index[df.Color2 == r['Color3']]
            if ind3.empty:
                ind3 = df.index[df.Color1 == r['Color3']]

        print(ind1[0], ind2[0], ind3[0])
        exit()
    data.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_validation_final_v3.csv', sep=";")

def plotDifference():
    data1 = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/simplified_RF_Data_v5.csv', sep=';',
                        keep_default_na=False)
    x_train, y_train, x_test_id, y_test = balancedTraining(data1)
    x_test = x_test_id.iloc[:, np.r_[0:5, 6:495]]
    print(x_test)
    path = '/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/cls_model_final_v2'
    if os.path.isfile(path):
        print("running improved model")
        with open(path, 'rb') as f:
            cls = cPickle.load(f)
    y_pred_test = cls.predict(x_test)
    minValues = [1, 3.25, 5.5, 7.75]
    maxValues = [3.25, 5.5, 7.75, 10.0]
    index = 0
    minDifferences = []
    for i in y_pred_test:
        minimum = minValues[int(i)-1]
        maximum = maxValues[int(i)-1]
        op = float(x_test_id['Average'].values.ravel()[index])
        diffMin = op - minimum
        diffMax = op - maximum
        if diffMax < 0 and diffMin > 0:
            minDifferences.append(0)
        if diffMax < 0 and diffMin < 0:
            minDifferences.append(min([abs(diffMin), abs(diffMax)]))
        if diffMax > 0 and diffMin > 0:
            minDifferences.append(min([abs(diffMin), abs(diffMax)]))
        index+=1

    fig, ax = plt.subplots(figsize=(10, 30))
    ax.hist(minDifferences, edgecolor='black', bins=[0.0,0.2,0.4,0.6,0.8,1.0,
                                                     1.2,1.4,1.6,1.8,2.0,
                                                     2.2,2.4,2.6,2.8,3.0,
                                                     3.2,3.4,3.6,3.8,4.0,
                                                     4.2,4.4,4.6,4.8,5.0,
                                                     5.2,5.4,5.6,5.8,6.0],align='left')
    ax.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0,
                     1.2,1.4,1.6,1.8,2.0,
                     2.2,2.4,2.6,2.8,3.0,
                     3.2,3.4,3.6,3.8,4.0,
                     4.2,4.4,4.6,4.8,5.0,
                     5.2,5.4,5.6,5.8,6.0])
    ax.set_xticklabels((0.0,0.2,0.4,0.6,0.8,1.0,
                             1.2,1.4,1.6,1.8,2.0,
                             2.2,2.4,2.6,2.8,3.0,
                             3.2,3.4,3.6,3.8,4.0,
                             4.2,4.4,4.6,4.8,5.0,
                             5.2,5.4,5.6,5.8,6.0))
    plt.xlabel("Difference")
    plt.ylabel("Count")
    plt.title("Difference between Class Value and Vote Average")
    # Show plot
    plt.show()

if __name__ == '__main__':
    plotDifference()
    # data = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_classifier.csv',
    #                     sep=';',
    #                     keep_default_na=False)
    # data2015 = data[data['post_create_date'].str.contains("2015")]
    # data2016 = data[data['post_create_date'].str.contains("2016")]
    # data2017 = data[data['post_create_date'].str.contains("2017")]
    # data2018 = data[data['post_create_date'].str.contains("2018")]
    # data2019 = data[data['post_create_date'].str.contains("2019")]
    # data2020 = data[data['post_create_date'].str.contains("2020")]
    # data2021 = data[data['post_create_date'].str.contains("2021")]
    # data2022 = data[data['post_create_date'].str.contains("2022")]
    # data2015.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_classifier_2015.csv', sep=";",
    #                  index=False)
    # data2016.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_classifier_2016.csv', sep=";",
    #                 index=False)
    # data2017.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_classifier_2017.csv', sep=";",
    #                 index=False)
    # data2018.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_classifier_2018.csv', sep=";",
    #                 index=False)
    # data2019.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_classifier_2019.csv', sep=";",
    #                 index=False)
    # data2020.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_classifier_2020.csv', sep=";",
    #                 index=False)
    # data2021.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_classifier_2021.csv', sep=";",
    #                 index=False)
    # data2022.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_classifier_2022.csv', sep=";",
    #                 index=False)
    # data = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_validation_final_v2.csv',
    #                    sep=';',
    #                    keep_default_na=False)
    # ColorEncoder(data)
    # data = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_validation_final_v3.csv',
    #                    sep=';',
    #                    keep_default_na=False)
    # results = predict_Random_forest(data)
    # results.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_validation.csv', sep=";",
    #                 index=False)
    # processes_pool = Pool(processes_count)
    # data = run_complex_operations(complex_operation, processes_pool)

    #showFeautreImportance(data1)
    #classDist(data1)
    #valueDist(data1)
    # data1 = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/simplified_RF_Data_v5.csv', sep=';',
    #                      keep_default_na=False)
    # x_train, y_train, x_test, y_test = balancedTraining(data1)
    # random_forest_balanced(x_train, y_train, x_test, y_test)
    # n_estimator = 2000
    # for i in range(3):
    #     print(i)
    #     x_train, y_train, x_test, y_test = balancedTraining(data1)
    #     retrain_Random_forest(x_train,y_train,x_test,y_test,i,n_estimator)
    #     n_estimator+=2000



