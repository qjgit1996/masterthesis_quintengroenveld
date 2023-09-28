from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst
import pandas as pd
import geopandas as gpd
import statistics

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

driver = gdal.GetDriverByName('GTiff')
filenameNoise = "/Users/quintengroenveld/qgis_masterthesis/road_multibuffer_modifiziert.tif" #path to raster
dataset_noise = gdal.Open(filenameNoise)
band = dataset_noise.GetRasterBand(1)
cols = dataset_noise.RasterXSize
rows = dataset_noise.RasterYSize
transform = dataset_noise.GetGeoTransform()
xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = -transform[5]
data = band.ReadAsArray(0, 0, cols, rows)
filenameEle = "/Users/quintengroenveld/qgis_masterthesis/elevation_reproj.tif" #path to raster
dataset_ele = gdal.Open(filenameEle)
band_ele = dataset_ele.GetRasterBand(1)
cols_ele = dataset_ele.RasterXSize
rows_ele = dataset_ele.RasterYSize
transform_ele = dataset_ele.GetGeoTransform()
xOrigin_ele = transform_ele[0]
yOrigin_ele = transform_ele[3]
pixelWidth_ele = transform_ele[1]
pixelHeight_ele = -transform_ele[5]
data_ele = band_ele.ReadAsArray(0, 0, cols_ele, rows_ele)
filenameLand = "/Users/quintengroenveld/qgis_masterthesis/landuse_final_reproj.tif" #path to raster
dataset_land = gdal.Open(filenameLand)
band_land = dataset_land.GetRasterBand(1)
cols_land = dataset_land.RasterXSize
rows_land = dataset_land.RasterYSize
transform_land = dataset_land.GetGeoTransform()
xOrigin_land = transform_land[0]
yOrigin_land = transform_land[3]
pixelWidth_land = transform_land[1]
pixelHeight_land = -transform_land[5]
data_land = band_land.ReadAsArray(0, 0, cols_land, rows_land)
shapefile1 = gpd.read_file("/Users/quintengroenveld/Documents/data_master_thesis/Keras-VGG16-places365/brit_data_v4.shp")

final_data = pd.DataFrame()
final_data = final_data.assign(id=shapefile1['ID'])
final_data = final_data.assign(Lat=shapefile1['Lat'])
final_data = final_data.assign(Lon=shapefile1['Lon'])
final_data = final_data.assign(Average=shapefile1['Average'])
final_data = final_data.assign(Variance=shapefile1['Variance'])
final_data = final_data.assign(Votes=shapefile1['Votes'])
final_data = final_data.assign(Geograph_U=shapefile1['Geograph U'])
final_data = final_data.assign(categories=shapefile1['categories'])
final_data = final_data.assign(Color1=shapefile1['color1'])
final_data = final_data.assign(Color2=shapefile1['color2'])
final_data = final_data.assign(Color3=shapefile1['color3'])
final_data = final_data.assign(Noise=0)
final_data = final_data.assign(Landuse=0)
final_data = final_data.assign(Elevation=0)

landuse_classes = list(range(1,45))

index = 0
for i in shapefile1['geometry']:
    # col = int((i.x - xOrigin) / pixelWidth)
    # row = int((yOrigin - i.y) / pixelHeight)
    # col_ele = int((i.x - xOrigin_ele) / pixelWidth_ele)
    # row_ele = int((yOrigin_ele - i.y) / pixelHeight_ele)
    # col_land = int((i.x - xOrigin_land) / pixelWidth_land)
    # row_land = int((yOrigin_land - i.y) / pixelHeight_land)
    # final_data['Noise'][index] = data[row][col]
    # final_data['Elevation'][index] = data_ele[row_ele][col_ele]
    # final_data['Landuse'][index] = data_land[row_land][col_land]
    # #print(final_data['Landuse'][index], data_land[row_land][col_land])
    # # print(statistics.median([data[row][col], data[row-1][col],data[row][col-1],data[row-1][col-1],data[row+1][col+1],data[row+1][col],data[row][col+1],data[row+1][col-1],data[row-1][col+1]]))
    # # break
    # index+=1
    col = int((i.x - xOrigin) / pixelWidth)
    row = int((yOrigin - i.y) / pixelHeight)
    col_ele = int((i.x - xOrigin_ele) / pixelWidth_ele)
    row_ele = int((yOrigin_ele - i.y) / pixelHeight_ele)
    col_land = int((i.x - xOrigin_land) / pixelWidth_land)
    row_land = int((yOrigin_land - i.y) / pixelHeight_land)
    final_data['Noise'][index] = statistics.median(
        [data[row][col],
         data[row-1][col],
         data[row][col-1],
         data[row-1][col-1],
         data[row+1][col+1],
         data[row+1][col],
         data[row][col+1],
         data[row+1][col-1],
         data[row-1][col+1]])
    final_data['Elevation'][index] = statistics.mean(
        [data_ele[row_ele][col_ele],
         data_ele[row_ele-1][col_ele],
         data_ele[row_ele][col_ele-1],
         data_ele[row_ele-1][col_ele-1],
         data_ele[row_ele+1][col_ele+1],
         data_ele[row_ele+1][col_ele],
         data_ele[row_ele][col_ele+1],
         data_ele[row_ele+1][col_ele-1],
         data_ele[row_ele-1][col_ele+1]])
    grid_landuse = [
        str(data_land[row_land][col_land]),
        str(data_land[row_land-1][col_land]),
        str(data_land[row_land][col_land-1]),
        str(data_land[row_land-1][col_land-1]),
        str(data_land[row_land+1][col_land+1]),
        str(data_land[row_land+1][col_land]),
        str(data_land[row_land][col_land+1]),
        str(data_land[row_land+1][col_land-1]),
        str(data_land[row_land-1][col_land+1])
    ]
    final_data['Landuse'][index] = ",".join(set(grid_landuse))
    # print(statistics.median([data[row][col], data[row-1][col],data[row][col-1],data[row-1][col-1],data[row+1][col+1],data[row+1][col],data[row][col+1],data[row+1][col-1],data[row-1][col+1]]))
    # break
    print(index)
    index += 1

final_data.to_csv('/Users/quintengroenveld/PycharmProjects/masterthesis/data_v4.csv', sep=";")



# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
#
# driver = gdal.GetDriverByName('GTiff')
# filenameNoise = "/Users/quintengroenveld/PycharmProjects/flickrHandler/road_multibuffer.tif" #path to raster
# dataset_noise = gdal.Open(filenameNoise)
# band = dataset_noise.GetRasterBand(1)
# cols = dataset_noise.RasterXSize
# rows = dataset_noise.RasterYSize
# transform = dataset_noise.GetGeoTransform()
# xOrigin = transform[0]
# yOrigin = transform[3]
# pixelWidth = transform[1]
# pixelHeight = -transform[5]
# data = band.ReadAsArray(0, 0, cols, rows)
# filenameEle = "/Users/quintengroenveld/swissmodel_masterthesis/elevation_final_final.tif" #path to raster
# dataset_ele = gdal.Open(filenameEle)
# band_ele = dataset_ele.GetRasterBand(1)
# cols_ele = dataset_ele.RasterXSize
# rows_ele = dataset_ele.RasterYSize
# transform_ele = dataset_ele.GetGeoTransform()
# xOrigin_ele = transform_ele[0]
# yOrigin_ele = transform_ele[3]
# pixelWidth_ele = transform_ele[1]
# pixelHeight_ele = -transform_ele[5]
# data_ele = band_ele.ReadAsArray(0, 0, cols_ele, rows_ele)
# filenameLand = "/Users/quintengroenveld/swissmodel_masterthesis/landuse_RF.tif" #path to raster
# dataset_land = gdal.Open(filenameLand)
# band_land = dataset_land.GetRasterBand(1)
# cols_land = dataset_land.RasterXSize
# rows_land = dataset_land.RasterYSize
# transform_land = dataset_land.GetGeoTransform()
# xOrigin_land = transform_land[0]
# yOrigin_land = transform_land[3]
# pixelWidth_land = transform_land[1]
# pixelHeight_land = -transform_land[5]
# data_land = band_land.ReadAsArray(0, 0, cols_land, rows_land)
# shapefile1 = gpd.read_file("/Users/quintengroenveld/swissmodel_masterthesis/flickrImages.shp")
#
# final_data = pd.DataFrame()
# final_data = final_data.assign(id=shapefile1['ID'])
# final_data = final_data.assign(Average=shapefile1['Average'])
# final_data = final_data.assign(Variance=shapefile1['Variance'])
# final_data = final_data.assign(Votes=shapefile1['Votes'])
# final_data = final_data.assign(categories=shapefile1['categories'])
# final_data = final_data.assign(Noise=0)
# final_data = final_data.assign(Landuse=0)
# final_data = final_data.assign(Elevation=0)
#
# landuse_classes = list(range(1,45))
#
# index = 0
# for i in shapefile1['geometry']:
#     # col = int((i.x - xOrigin) / pixelWidth)
#     # row = int((yOrigin - i.y) / pixelHeight)
#     # col_ele = int((i.x - xOrigin_ele) / pixelWidth_ele)
#     # row_ele = int((yOrigin_ele - i.y) / pixelHeight_ele)
#     # col_land = int((i.x - xOrigin_land) / pixelWidth_land)
#     # row_land = int((yOrigin_land - i.y) / pixelHeight_land)
#     # final_data['Noise'][index] = data[row][col]
#     # final_data['Elevation'][index] = data_ele[row_ele][col_ele]
#     # final_data['Landuse'][index] = data_land[row_land][col_land]
#     # #print(final_data['Landuse'][index], data_land[row_land][col_land])
#     # # print(statistics.median([data[row][col], data[row-1][col],data[row][col-1],data[row-1][col-1],data[row+1][col+1],data[row+1][col],data[row][col+1],data[row+1][col-1],data[row-1][col+1]]))
#     # # break
#     # index+=1
#     col = int((i.x - xOrigin) / pixelWidth)
#     row = int((yOrigin - i.y) / pixelHeight)
#     col_ele = int((i.x - xOrigin_ele) / pixelWidth_ele)
#     row_ele = int((yOrigin_ele - i.y) / pixelHeight_ele)
#     col_land = int((i.x - xOrigin_land) / pixelWidth_land)
#     row_land = int((yOrigin_land - i.y) / pixelHeight_land)
#     final_data['Noise'][index] = statistics.median(
#         [data[row][col],
#          data[row-1][col],
#          data[row][col-1],
#          data[row-1][col-1],
#          data[row+1][col+1],
#          data[row+1][col],
#          data[row][col+1],
#          data[row+1][col-1],
#          data[row-1][col+1]])
#     final_data['Elevation'][index] = statistics.mean(
#         [data_ele[row_ele][col_ele],
#          data_ele[row_ele-1][col_ele],
#          data_ele[row_ele][col_ele-1],
#          data_ele[row_ele-1][col_ele-1],
#          data_ele[row_ele+1][col_ele+1],
#          data_ele[row_ele+1][col_ele],
#          data_ele[row_ele][col_ele+1],
#          data_ele[row_ele+1][col_ele-1],
#          data_ele[row_ele-1][col_ele+1]])
#     grid_landuse = [
#         str(data_land[row_land][col_land]),
#         str(data_land[row_land-1][col_land]),
#         str(data_land[row_land][col_land-1]),
#         str(data_land[row_land-1][col_land-1]),
#         str(data_land[row_land+1][col_land+1]),
#         str(data_land[row_land+1][col_land]),
#         str(data_land[row_land][col_land+1]),
#         str(data_land[row_land+1][col_land-1]),
#         str(data_land[row_land-1][col_land+1])
#     ]
#     final_data['Landuse'][index] = ",".join(set(grid_landuse))
#     # print(statistics.median([data[row][col], data[row-1][col],data[row][col-1],data[row-1][col-1],data[row+1][col+1],data[row+1][col],data[row][col+1],data[row+1][col-1],data[row-1][col+1]]))
#     # break
#     print(index)
#     index += 1
#
# final_data.to_csv('/Users/quintengroenveld/PycharmProjects/flickrHandler/swiss_data_v1.csv', sep=";")