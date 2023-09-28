# import cv2
# from matplotlib import pyplot as plt
#
# img1 = cv2.imread('/Users/quintengroenveld/qgis_masterthesis/road30m.tif', -1)
# img2 = cv2.imread('/Users/quintengroenveld/qgis_masterthesis/road60m.tif', -1)
# img3 = cv2.imread('/Users/quintengroenveld/qgis_masterthesis/road120m.tif', -1)
# img4 = cv2.imread('/Users/quintengroenveld/qgis_masterthesis/road240m.tif', -1)
# img5 = cv2.imread('/Users/quintengroenveld/qgis_masterthesis/road480m.tif', -1)
# img6 = cv2.imread('/Users/quintengroenveld/qgis_masterthesis/road960m.tif', -1)
# img7 = cv2.imread('/Users/quintengroenveld/qgis_masterthesis/road1920m.tif', -1)
# img8 = cv2.imread('/Users/quintengroenveld/qgis_masterthesis/road3840m.tif', -1)
#
# dst = cv2.add(img1,img2)
# dst1 = cv2.add(img3,img4)
# dst2 = cv2.add(img5,img6)
# dst3 = cv2.add(img7,img8)
#
# dst4 = cv2.add(dst, dst1)
# dst5 = cv2.add(dst2, dst3)
# dst6 = cv2.add(dst4, dst5)
#
# # histg = cv2.calcHist([dst6],[0],None,[256],[0,256])
# # plt.plot(histg)
# # plt.show()
# cv2.imwrite("road_multibuffer.tif", dst6)
# cv2.imshow('dst6',dst6)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import geopandas as gpd
#
# shapefile = gpd.read_file("/Users/quintengroenveld/qgis_masterthesis/test_data_landuse.tif")
# shp = shapefile.set_crs(3035)
# # change CRS to epsg 4326
# data = shp.to_crs(epsg=27700)
# # write shp file
# data.to_file('/Users/quintengroenveld/qgis_masterthesis/test_data_landuse_final.tif')
import rasterio
import rioxarray

rds = rioxarray.open_rasterio('/Users/quintengroenveld/qgis_masterthesis/landuse_final_raster.tif')
crs = rasterio.crs.CRS({"init": "epsg:3035"})
rds.crs = crs
rds_27700 = rds.rio.reproject("EPSG:27700")
rds_27700.rio.to_raster("/Users/quintengroenveld/qgis_masterthesis/landuse_final.tif")