import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve, LearningCurveDisplay
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

from categoriesList import cat

from landuseList import land
import math
import numpy as np
from multiprocessing import Pool
from sklearn import metrics
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import _pickle as cPickle

from objectList import objects

# hand tensor (array) dem RF
# objectdetektionsmodell integrieren?

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def complex_operation(input):
    categoriesMatrix = np.array([])
    landuseMatrix = np.array([])
    objectMatrix = np.array([])
    landuseCats = list(map(str, [111.0,112.0,121.0,122.0,123.0,124.0,131.0,132.0,133.0,141.0,142.0,211.0,212.0,213.0,221.0,222.0,223.0,231.0,241.0,242.0,243.0,244.0,311.0,312.0,313.0,321.0,322.0,323.0,324.0,331.0,332.0,333.0,334.0,335.0,411.0,412.0,421.0,422.0,423.0,511.0,512.0,521.0,522.0,523.0]))
    cocoClasses = list(map(str, range(1, 81)))
    index = 0
    for i, r in input.iterrows():
        landuse = r["Landuse"].split(",")
        categories = r["categories"].split(",")
        objects = r["Objects"]
        mask = np.isin(cat, categories)
        maskLanduse = np.isin(landuseCats, landuse)
        maskObjects = np.isin(cocoClasses, objects)
        if index == 0:
            categoriesMatrix = [mask.astype(int)]
            landuseMatrix = [maskLanduse.astype(int)]
            objectMatrix = [maskObjects.astype(int)]

        else:
            categoriesMatrix = np.append(categoriesMatrix, [mask.astype(int)], axis=0)
            landuseMatrix = np.append(landuseMatrix, [maskLanduse.astype(int)], axis=0)
            objectMatrix = np.append(objectMatrix, [maskObjects.astype(int)], axis=0)
        index += 1
        # list_categories =np.asarray(list(r["categories"]), dtype=np.float64)
        # list_landuse = np.asarray(list(r["Landuse"]), dtype=np.float64)
        # data. = list_categories
        # data.at[i, "Landuse"] = list_landuse
    # Concatenate the encoded variables with the original numerical variables

    return categoriesMatrix, landuseMatrix, objectMatrix
    # print(df_new)
    # exit()
    # x = np.asarray(input_data[['Noise', 'Elevation', "Landuse"]])
    #
    #
    # y = input_data.iloc[:, 2].values.ravel()
    #
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    # regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    # regressor.fit(X_train, y_train)
    # y_pred_test = regressor.predict(X_test)
    # accuracy = r2_score(y_test, y_pred_test)
    # tree.plot_tree(regressor.estimators_[0], max_depth=5, fontsize=5)
    # plt.show()
    # print( 'Linear Regression Accuracy: ', accuracy*100,'%')


def run_complex_operations(operation, pool):
    df = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_validation_ob_land_cat.csv', sep=';', keep_default_na=False)
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
    df_new2 = pd.concat([pd.DataFrame(x[0][2]), pd.DataFrame(x[1][2]), pd.DataFrame(x[2][2]), pd.DataFrame(x[3][2]),
                         pd.DataFrame(x[4][2]), pd.DataFrame(x[5][2]), pd.DataFrame(x[6][2]), pd.DataFrame(x[7][2]),
                         pd.DataFrame(x[8][2]), pd.DataFrame(x[9][2])], axis=0)
    df_new = df_new.reset_index()
    df_new1 = df_new1.reset_index()
    df_new2 = df_new2.reset_index()
    df = df.reset_index()
    df_new = pd.concat([df[['id', 'x', 'y', 'link', 'Color1', 'Color2', 'Color3', 'Noise', 'Elevation']], df_new, df_new1, df_new2], axis=1)
    df_new.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_validation_data.csv', sep=";", index=False)
    return df_new


def random_forest(data):
    x = data.iloc[:, np.r_[0:2, 4:369, 370:414]]
    y = data.iloc[:, 2].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    regressor = RandomForestRegressor(n_estimators=600,
                                      max_depth=100,
                                      min_samples_split=59,
                                      max_leaf_nodes=186,
                                      min_samples_leaf=4,
                                      max_samples=0.1,
                                      # max_features=131,
                                      random_state=0,
                                      warm_start=True)
    regressor.fit(X_train, y_train)
    y_pred_test = regressor.predict(X_test)
    lossData = pd.read_csv('/Models_and_plots/loss_v2.csv', sep=';',
                           keep_default_na=False)
    lossData.loc[len(lossData.index)] = [500, metrics.r2_score(y_test, y_pred_test),
                                         metrics.r2_score(y_test, y_pred_test),
                                         metrics.explained_variance_score(y_test, y_pred_test),
                                         metrics.mean_squared_error(y_test, y_pred_test, squared=False)]
    lossData.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/loss_v2.csv', sep=";", index=False)
    with open('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/model_n' + '600', 'wb') as f:
        cPickle.dump(regressor, f)
    return y_pred_test, y_test


def retrainModel(data, estimatorToAdd):
    x = data.iloc[:, np.r_[0:2, 4:369, 370:414]]
    x.columns = x.columns.astype(str)
    y = data.iloc[:, 2].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    lossData = pd.read_csv('/Models_and_plots/loss.csv', sep=';',
                           keep_default_na=False)
    with open(str('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/model_n' + str(
            int(lossData.iloc[-1]['n_of_trees']))), 'rb') as f:
        regressor = cPickle.load(f)
        regressor.set_params(n_estimators=estimatorToAdd,
                             max_depth=100,
                             min_samples_split=59,
                             max_leaf_nodes=186,
                             min_samples_leaf=4,
                             max_samples=0.1,
                             max_features=131,
                             random_state=0,
                             warm_start=True)
        regressor.fit(X_train, y_train)
        y_pred_test = regressor.predict(X_test)
        lossData.loc[len(lossData.index)] = [estimatorToAdd, metrics.r2_score(y_test, y_pred_test),
                                             metrics.r2_score(y_test, y_pred_test),
                                             metrics.explained_variance_score(y_test, y_pred_test),
                                             metrics.mean_squared_error(y_test, y_pred_test, squared=False)]
    lossData.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/loss.csv', sep=";", index=False)
    train_sizes, train_scores, test_scores = learning_curve(regressor, x, y)
    display = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores,
                                   score_name="Score")
    display.plot()
    plt.show()
    with open(str('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/model_n' + str(
            int(lossData.iloc[-1]['n_of_trees']))), 'wb') as f:
        cPickle.dump(regressor, f)


def plotModel(data, model):
    x = data.iloc[:, np.r_[0:2, 4:369, 370:414]]
    x.columns = x.columns.astype(str)
    y = data.iloc[:, 2].values.ravel()
    with open(str('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/' + model), 'rb') as f:
        regressor = cPickle.load(f)
        train_sizes, train_scores, test_scores = learning_curve(regressor, x, y)
        display = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores,
                                       score_name="Score")
        display.plot()
        plt.show()


def maxDepthSearch():
    x = data.iloc[:, np.r_[0:2, 4:369, 370:414]]
    x.columns = x.columns.astype(str)
    y = data.iloc[:, 2].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    # define lists to collect scores
    train_scores, test_scores = list(), list()
    # define the tree depths to evaluate
    values = [i for i in range(15, 30, 1)]
    # evaluate a decision tree for each depth
    for i in values:
        model = RandomForestRegressor(n_estimators=40, max_depth=i, random_state=0, warm_start=True)
        model.fit(X_train, y_train)
        # evaluate on the train dataset
        train_yhat = model.predict(X_train)
        train_acc = metrics.r2_score(y_train, train_yhat)
        train_scores.append(train_acc)
        # evaluate on the test dataset
        test_yhat = model.predict(X_test)
        test_acc = metrics.r2_score(y_test, test_yhat)
        test_scores.append(test_acc)
        # summarize progress
        print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
    # plot of train and test scores vs tree depth
    pyplot.plot(values, train_scores, '-o', label='Train')
    pyplot.plot(values, test_scores, '-o', label='Test')
    pyplot.legend()
    pyplot.show()


def minSampleSplitSearch(data):
    x = data.iloc[:, np.r_[0:2, 4:369, 370:414]]
    x.columns = x.columns.astype(str)
    y = data.iloc[:, 2].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    # define lists to collect scores
    train_scores, test_scores = list(), list()
    # define the tree depths to evaluate
    values = [i for i in range(1, 100, 2)]
    # evaluate a decision tree for each depth
    for i in values:
        model = RandomForestRegressor(n_estimators=40, max_depth=21, min_samples_split=i, random_state=0,
                                      warm_start=True)
        model.fit(X_train, y_train)
        # evaluate on the train dataset
        train_yhat = model.predict(X_train)
        train_acc = metrics.r2_score(y_train, train_yhat)
        train_scores.append(train_acc)
        # evaluate on the test dataset
        test_yhat = model.predict(X_test)
        test_acc = metrics.r2_score(y_test, test_yhat)
        test_scores.append(test_acc)
        # summarize progress
        print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
    # plot of train and test scores vs tree depth
    pyplot.plot(values, train_scores, '-o', label='Train')
    pyplot.plot(values, test_scores, '-o', label='Test')
    pyplot.legend()
    pyplot.show()


def maxLeafNodesSearch(data):
    x = data.iloc[:, np.r_[0:2, 4:369, 370:414]]
    x.columns = x.columns.astype(str)
    y = data.iloc[:, 2].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    # define lists to collect scores
    train_scores, test_scores = list(), list()
    # define the tree depths to evaluate
    values = [i for i in range(2, 200, 2)]
    # evaluate a decision tree for each depth
    for i in values:
        model = RandomForestRegressor(n_estimators=40,
                                      max_depth=21,
                                      min_samples_split=59,
                                      max_leaf_nodes=i,
                                      random_state=0,
                                      warm_start=True)
        model.fit(X_train, y_train)
        # evaluate on the train dataset
        train_yhat = model.predict(X_train)
        train_acc = metrics.r2_score(y_train, train_yhat)
        train_scores.append(train_acc)
        # evaluate on the test dataset
        test_yhat = model.predict(X_test)
        test_acc = metrics.r2_score(y_test, test_yhat)
        test_scores.append(test_acc)
        # summarize progress
        print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
    # plot of train and test scores vs tree depth
    pyplot.plot(values, train_scores, '-o', label='Train')
    pyplot.plot(values, test_scores, '-o', label='Test')
    pyplot.legend()
    pyplot.show()


def minSamplesLeafSearch(data):
    x = data.iloc[:, np.r_[0:2, 4:369, 370:414]]
    x.columns = x.columns.astype(str)
    y = data.iloc[:, 2].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    # define lists to collect scores
    train_scores, test_scores = list(), list()
    # define the tree depths to evaluate
    values = [i for i in range(1, 20, 2)]
    # evaluate a decision tree for each depth
    for i in values:
        model = RandomForestRegressor(n_estimators=40,
                                      max_depth=21,
                                      min_samples_split=59,
                                      max_leaf_nodes=186,
                                      min_samples_leaf=i,
                                      random_state=0,
                                      warm_start=True)
        model.fit(X_train, y_train)
        # evaluate on the train dataset
        train_yhat = model.predict(X_train)
        train_acc = metrics.r2_score(y_train, train_yhat)
        train_scores.append(train_acc)
        # evaluate on the test dataset
        test_yhat = model.predict(X_test)
        test_acc = metrics.r2_score(y_test, test_yhat)
        test_scores.append(test_acc)
        # summarize progress
        print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
    # plot of train and test scores vs tree depth
    pyplot.plot(values, train_scores, '-o', label='Train')
    pyplot.plot(values, test_scores, '-o', label='Test')
    pyplot.legend()
    pyplot.show()


def maxSamplesSearch(data):
    x = data.iloc[:, np.r_[0:2, 4:369, 370:414]]
    x.columns = x.columns.astype(str)
    y = data.iloc[:, 2].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    # define lists to collect scores
    train_scores, test_scores = list(), list()
    # define the tree depths to evaluate
    values = np.arange(0.1, 1.0, 0.1)
    # evaluate a decision tree for each depth
    for i in values:
        model = RandomForestRegressor(n_estimators=40,
                                      max_depth=21,
                                      min_samples_split=59,
                                      max_leaf_nodes=186,
                                      min_samples_leaf=4,
                                      max_samples=i,
                                      random_state=0,
                                      warm_start=True)
        model.fit(X_train, y_train)
        # evaluate on the train dataset
        train_yhat = model.predict(X_train)
        train_acc = metrics.r2_score(y_train, train_yhat)
        train_scores.append(train_acc)
        # evaluate on the test dataset
        test_yhat = model.predict(X_test)
        test_acc = metrics.r2_score(y_test, test_yhat)
        test_scores.append(test_acc)
        # summarize progress
        print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
    # plot of train and test scores vs tree depth
    pyplot.plot(values, train_scores, '-o', label='Train')
    pyplot.plot(values, test_scores, '-o', label='Test')
    pyplot.legend()
    pyplot.show()


def maxFeaturesSearch(data):
    x = data.iloc[:, np.r_[0:2, 4:369, 370:414]]
    x.columns = x.columns.astype(str)
    y = data.iloc[:, 2].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    # define lists to collect scores
    train_scores, test_scores = list(), list()
    # define the tree depths to evaluate
    values = [i for i in range(1, 400, 2)]
    # evaluate a decision tree for each depth
    for i in values:
        model = RandomForestRegressor(n_estimators=40,
                                      max_depth=21,
                                      min_samples_split=59,
                                      max_leaf_nodes=186,
                                      min_samples_leaf=4,
                                      max_samples=0.1,
                                      max_features=i,
                                      random_state=0,
                                      warm_start=True)
        model.fit(X_train, y_train)
        # evaluate on the train dataset
        train_yhat = model.predict(X_train)
        train_acc = metrics.r2_score(y_train, train_yhat)
        train_scores.append(train_acc)
        # evaluate on the test dataset
        test_yhat = model.predict(X_test)
        test_acc = metrics.r2_score(y_test, test_yhat)
        test_scores.append(test_acc)
        # summarize progress
        print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
    # plot of train and test scores vs tree depth
    pyplot.plot(values, train_scores, '-o', label='Train')
    pyplot.plot(values, test_scores, '-o', label='Test')
    pyplot.legend()
    pyplot.show()


def showFeautreImportance(data):
    x = data.iloc[:, np.r_[0:2, 4:369, 370:414]]
    x.columns = x.columns.astype(str)
    y = data.iloc[:, 2].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    with open('/Models_and_plots/model_n500', 'rb') as f:
        regressor = cPickle.load(f)
    result = permutation_importance(
        regressor, X_test, y_test, n_repeats=10, random_state=1, n_jobs=2
    )

    forest_importances = pd.Series(result.importances_mean, index=x.columns)
    forest_importances.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/featureImportance.csv', sep=";")
    # subsets = [forest_importances[0:(len(forest_importances)/4)],
    #            forest_importances[(len(forest_importances)//4):(len(forest_importances)//2)],
    #             forest_importances[(len(forest_importances)//2):(len(forest_importances)//2+len(forest_importances)//4)],
    #             forest_importances[len((forest_importances)//2+len(forest_importances)//4):]]
    # index = 1
    # for i in subsets:
    #     fig, ax = plt.subplots()
    #     i.plot.bar(yerr=result.importances_std, ax=ax)
    #     ax.set_title("Feature importances using permutation on full model section " + index)
    #     ax.set_ylabel("Mean accuracy decrease")
    #     ax.set_ylim(0, 0.5)
    #     fig.tight_layout()
    #     plt.savefig(str('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/figureSection' + index + '.jpg'))
    #     plt.show()


def histVarianceSoN(data):
    # Creating histogram
    x = np.array(data['Variance'])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(x, bins=[0,1,2,3,4,5,6,7,8,9,10])
    plt.xlabel("Variance")
    plt.ylabel("Count")
    plt.title('Variance inside of the Scenic-or-Not Votes')
    # Show plot
    plt.show()

def histVarianceModel(y_pred_test, y_test):
    # Creating histogram
    variance = (y_pred_test-y_test)**2
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(variance, bins=[0,1,2,3,4,5,6,7,8,9,10])
    plt.xlabel("Variance")
    plt.ylabel("Count")
    plt.title('Variance inside of the test predictions')
    # Show plot
    plt.show()

def simplifyData(data):
    data[['Color1', 'Color2', 'Color3']] = data[['Color1', 'Color2', 'Color3']].apply(LabelEncoder().fit_transform)
    data = data.assign(SimpleValue=0)
    data.loc[data['Average'] > 0.999, "SimpleValue"] = 1
    data.loc[data['Average'] > 3.25, "SimpleValue"] = 2
    data.loc[data['Average'] >= 5.5, "SimpleValue"] = 3
    data.loc[data['Average'] >= 7.75, "SimpleValue"] = 4
    data.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/simplified_RF_Data_v5.csv', sep=";")

def predict_Random_forest(x):
    Xdata = x.iloc[:, np.r_[5:7, 8:373, 373:417]]
    path = '/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/model_n600'
    print("running improved model")
    with open('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/model_n600', 'rb') as f:
        regressor = cPickle.load(f)
    y_pred = regressor.predict(Xdata)
    x['Predictions'] = y_pred.tolist()
    return x

processes_count = 10

if __name__ == '__main__':
    # data = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/finaldata_for_RF_v2.csv',
    #                      sep=';', keep_default_na=False)
    # simplifyData(data)
    processes_count = 10
    processes_pool = Pool(processes_count)
    data = run_complex_operations(complex_operation, processes_pool)
    data.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_validation_final.csv', sep=";", index=False)
    # n_e = 420
    data = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_validation_final.csv', sep=';', keep_default_na=False)
    # # y_pred_test, y_test = random_forest(data)
    # # histVarianceSoN(data)
    # # histVarianceModel(y_pred_test, y_test)
    # # simplifyData(data1)
    # # showFeautreImportance(data)
    dict = {}
    indexCat = 0
    indexLand = 0
    indexObj = 0
    del data['index']
    del data['index.1']
    del data['index.2']
    for i in range(0, len(data.columns)):
        if i < 9:
            dict[data.columns[i]] = data.columns[i]
            continue
        elif i >= 9 and i < 374:
            dict[data.columns[i]] = cat[indexCat]
            indexCat+=1
        elif i >= 374 and i < 418:
            dict[data.columns[i]] = land[indexLand]
            indexLand+=1
        elif i >= 418:
            dict[data.columns[i]] = objects[indexObj]
            indexObj+=1
    data1 = data.rename(columns=dict)
    data1.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_validation_final_v2.csv', sep=";", index=False)

    # [0:2, 4:369, 370:414]
    # # # for i in range(1):
    # # #     retrainModel(data, n_e)
    # # #     n_e+=20
    # retrainModel(data, 500)
    # plotModel(data=data, model="model_n500")
    # lossData = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/Models_and_plots/loss.csv', sep=';',keep_default_na=False)
    # plt.plot(lossData['loss'])
    # plt.show()
    # maxDepthSearch(data)
    # minSampleSplitSearch(data)
    # maxLeafNodesSearch(data)
    # minSamplesLeafSearch(data)
    # maxSamplesSearch(data)
    # maxFeaturesSearch(data)

    # data = pd.read_csv('/Users/quintengroenveld/PycharmProjects/masterthesis2/Swiss_finaldata_for_RF_v1_1.csv', sep=';',
    #                    keep_default_na=False)
    # results = predict_Random_forest(data)
    # results.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/swiss_results_regressor.csv', sep=";",
    #              index=False)

