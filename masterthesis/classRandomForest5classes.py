import os

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, LearningCurveDisplay, RandomizedSearchCV
from sklearn.inspection import permutation_importance

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

def random_forest(data):
    x = data.iloc[:, np.r_[1:3, 5:370, 371:415]]
    y = data.iloc[:, 415].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    cls = RandomForestClassifier(n_estimators= 2000, min_samples_split=10, min_samples_leaf= 50, max_leaf_nodes=100, max_depth=20, class_weight='balanced',
                                      random_state=0,
                                      warm_start=True)
    cls.fit(X_train, y_train)
    y_pred_test = cls.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    print("F1 score: " + str(f1_score(y_test, y_pred_test, average=None)))

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0.999-3.0", "3.0-5.0", "5.0-6.0", "6.0-8.0", "8.0-10.0"]).plot()
    pyplot.show()

def searchParam(data):
    x = data.iloc[:, np.r_[1:3, 5:370, 371:415]]
    y = data.iloc[:, 415].values.ravel()
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

if __name__ == '__main__':
    data1 = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/simplified_RF_Data.csv', sep=';',
                       keep_default_na=False)
    #searchParam(data1)
    random_forest(data1)