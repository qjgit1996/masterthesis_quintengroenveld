from osgeo import gdal

driver = gdal.GetDriverByName('GTiff')
filename = "/Users/quintengroenveld/qgis_masterthesis/swiss_results_2016_interpolated_clipped.tif" #path to raster
dataset = gdal.Open(filename)

for i in dataset:
    print(i)