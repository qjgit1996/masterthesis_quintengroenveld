import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import numpy as np
import skgstat as skg
import pyproj
from pyproj import CRS

def semivariogramm(data):
    wgs84 = CRS.from_epsg(4326)
    epsg2056 = CRS.from_epsg(2056)
    df = data
    #.sample(n=10000, random_state=0)
    #df.to_csv('/Users/quintengroenveld/PycharmProjects/discussion/testsubsample.csv', sep=";")
    x1, y1 = pyproj.transform(wgs84, epsg2056, df['latitude'].ravel(), df['longitude'].ravel())
    coords = np.c_[x1, y1]
    values = df['Prediction'].ravel()
    with np.errstate(divide='ignore', invalid='ignore'):
        V = skg.Variogram(coords, values, n_lags = 200, bin_func = 'even', maxlag = 2000)
        fig, axes = plt.subplots(1, 1, sharey=True)
        axes.plot(V.bins, V.experimental, '.b')
        axes.set_ylim([0, 1.0])
        axes.set_xlabel('Lag in [m]')
        axes.set_ylabel('Semivariance')
        plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
     data = pd.read_csv('/Users/quintengroenveld/qgis_masterthesis/semivariogrammPunkte.csv',sep=';',keep_default_na=False)
     semivariogramm(data)