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

def random_forest_balanced(X_train_id, y_train, X_test_id, y_test):
    X_train = X_train_id[:, 1:]
    X_test = X_test_id.iloc[:, 1:]
    cls = RandomForestClassifier(n_estimators= 2000, min_samples_split=10, min_samples_leaf= 5, max_leaf_nodes=200, max_depth=10, class_weight=None,
                                      random_state=0,
                                      warm_start=True)
    cls.fit(X_train, y_train)
    y_pred_test = cls.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    print("F1 score: " + str(f1_score(y_test, y_pred_test, average=None)))

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0.999-5.5", "5.5-10.0"]).plot()
    pyplot.show()

    with open('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/dual_cls_model', 'wb') as f:
        cPickle.dump(cls, f)
    return y_pred_test, y_test, X_test_id

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


def balancedTraining(data):
    x = data.iloc[:, np.r_[7:12, 13:503]]
    y = data.iloc[:, np.r_[503]]
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

def predict_Random_forest(x):
    Xdata = x.iloc[:, np.r_[8:13, 14:378, 378:422, 422:503]]
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
    data = data.assign(SimpleValue=0)
    data.loc[data['Average'] > 0.999, "SimpleValue"] = 1
    data.loc[data['Average'] >= 5.5, "SimpleValue"] = 2
    data.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/simplified_RF_Data_v7.csv', sep=";")

if __name__ == '__main__':
    data = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/simplified_RF_Data_v7.csv',
                        sep=';',
                        keep_default_na=False)
    #simplifyData(data)
    x_train, y_train, x_test, y_test = balancedTraining(data)
    random_forest_balanced(x_train, y_train, x_test, y_test)