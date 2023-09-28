import time
from multiprocessing import Pool

import pandas as pd
import os
import datetime
import requests
import numpy as np
from pandas import DataFrame

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def complex_operation(input):
    index = 0
    data = input.loc[input['post_create_date'].dt.year > 2015]
    # for i, r in input.iterrows():
    #     print(r,index)
    #     date = r['post_create_date'].split(" ")
    #     date_object = datetime.datetime.strptime(date[0], '%Y-%m-%d').date()
    #     if date_object.year <= 2015:
    #         input = input.drop(input.index[[index]])
    return data


def run_complex_operations(operation, pool):
    df = pd.read_csv('flickr_images.csv', sep=',')
    df['post_create_date'] = pd.to_datetime(df['post_create_date'], errors='coerce')
    df['post_create_date'].fillna(value='2015-1-1 00:00:00', inplace=True)
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
    df_new = pd.concat([pd.DataFrame(x[0]), pd.DataFrame(x[1]), pd.DataFrame(x[2]), pd.DataFrame(x[3]),
                        pd.DataFrame(x[4]), pd.DataFrame(x[5]), pd.DataFrame(x[6]), pd.DataFrame(x[7]),
                        pd.DataFrame(x[8]), pd.DataFrame(x[9])], axis=0)
    df_new = df_new.reset_index()
    return df_new


def loadData(data):
    indexImage = 0
    data1 = data.iloc[600318:]
    for i, r in data1.iterrows():
        try:
            date = r['post_create_date'].split(" ")
            date_object = datetime.datetime.strptime(date[0], '%Y-%m-%d').date()
            if date_object.year > 2015:
                if not os.path.exists("/Users/quintengroenveld/Documents/data_master_thesis/flickrImages/" + str(date_object.year)+"/"+str(date_object.month)):
                    os.makedirs("/Users/quintengroenveld/Documents/data_master_thesis/flickrImages/"+str(date_object.year)+"/"+str(date_object.month))
                img_data = requests.get(r['post_thumbnail_url']).content
                with open("/Users/quintengroenveld/Documents/data_master_thesis/flickrImages/"+str(date_object.year)+"/"+str(date_object.month)+ "/" + r['post_guid']+'.jpg', 'wb') as handler:
                    handler.write(img_data)
                indexImage+=1
        except:
            print("An error occurred")
        print(i)
        if indexImage%10000 == 0 and i!=0:
            time.sleep(5 * 60)


def rearangePhotos(data):
    for i, r in data.iterrows():
        try:
            date = r['post_create_date'].split(" ")
            date_object = datetime.datetime.strptime(date[0], '%Y-%m-%d').date()
            if date_object.year <= 2015:
                data = data.drop(data.index[[i]])
        except:
            print("An error occurred")
    return data


if __name__ == '__main__':
    processes_count = 10
    #data = pd.read_csv('flickr_images.csv', sep=',', keep_default_na=False)
    processes_pool = Pool(processes_count)
    data = run_complex_operations(complex_operation, processes_pool)
    #loadData(data)
    #newData = rearangePhotos(data)
    data.to_csv('/Users/quintengroenveld/PycharmProjects/flickrHandler/swissFlickrImg2016_2023.csv', sep=";", index=False)

