from bs4 import BeautifulSoup
import requests
import geopandas as gpd
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

index = 0
shapefile1 = gpd.read_file("/Users/quintengroenveld/qgis_masterthesis/votes_data.shp")
for i in shapefile1['Geograph U']:
    try:
        URL = i # Replace this with the website's URL
        getURL = requests.get(URL, headers={"User-Agent":"Mozilla/5.0"})
        soup = BeautifulSoup(getURL.text, 'html.parser')
        images = soup.find_all('img')
        resolvedURLs = []
        src = images[0].get('src')
        resolvedURLs.append(requests.compat.urljoin(URL, src))
        shapefile1['Geograph U'][index] = resolvedURLs[0]
        index+=1
        print(index)
            # webs = requests.get(image)
            # open('images/' + image.split('/')[-1], 'wb').write(webs.content)
    except:
        print("An exception")

shapefile1.to_file("/Users/quintengroenveld/qgis_masterthesis/votes_updated.shp")