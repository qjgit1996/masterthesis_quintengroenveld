# import pandas as pd
# import math
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd
#
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
#
# data = pd.read_csv('data_v2.csv', sep=';', keep_default_na=False)
# for i, r in data.iterrows():
#     # data.at[i, "Average"] = int(round(r["Average"],0))
#     if r["categories"] == "":
#
#         data = data.drop(i, axis=0)
#         # print(data.drop(i, axis=0))
#
# label_encoder = LabelEncoder()
# label_encoded = label_encoder.fit_transform(data[['categories']])
# label_encoder1 = LabelEncoder()
# label_encoded1 = label_encoder1.fit_transform(data[['Landuse']])
# # Concatenate the encoded variables with the original numerical variables
# data = data.reset_index(drop=True)
# X = pd.merge(pd.DataFrame(label_encoded), data[['id', 'Average', "Variance", "Votes", "Noise", "Elevation"]], left_index=True, right_index=True)
# X_complete = pd.merge(pd.DataFrame(label_encoded1), X, left_index=True, right_index=True)
# X_complete.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/data_v2_final.csv', sep=";")
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv('data_v2.csv', sep=';', keep_default_na=False)
for i, r in data.iterrows():
    # data.at[i, "Average"] = int(round(r["Average"],0))
    if r["categories"] == "":
        data = data.drop(i, axis=0)
        # print(data.drop(i, axis=0))

# Concatenate the encoded variables with the original numerical variables
data = data.reset_index(drop=True)
data.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/data_v2_final.csv', sep=";")

