import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('/Users/quintengroenveld/swissmodel_masterthesis/swiss_roadrail_30m_extracted.tif', -1)
img2 = cv2.imread('/Users/quintengroenveld/swissmodel_masterthesis/swiss_roadrail_60m_extracted.tif', -1)
img3 = cv2.imread('/Users/quintengroenveld/swissmodel_masterthesis/swiss_roadrail_120m_extracted.tif', -1)
img4 = cv2.imread('/Users/quintengroenveld/swissmodel_masterthesis/swiss_roadrail_240m_extracted.tif', -1)
img5 = cv2.imread('/Users/quintengroenveld/swissmodel_masterthesis/swiss_roadrail_480m_extracted.tif', -1)
img6 = cv2.imread('/Users/quintengroenveld/swissmodel_masterthesis/swiss_roadrail_960m_extracted.tif', -1)
img7 = cv2.imread('/Users/quintengroenveld/swissmodel_masterthesis/swiss_roadrail_1920m_extracted.tif', -1)
img8 = cv2.imread('/Users/quintengroenveld/swissmodel_masterthesis/swiss_roadrail_3840m_extracted.tif', -1)

dst = cv2.add(img1,img2)
dst1 = cv2.add(img3,img4)
dst2 = cv2.add(img5,img6)
dst3 = cv2.add(img7,img8)

dst4 = cv2.add(dst, dst1)
dst5 = cv2.add(dst2, dst3)
dst6 = cv2.add(dst4, dst5)

# histg = cv2.calcHist([dst6],[0],None,[256],[0,256])
# plt.plot(histg)
# plt.show()
cv2.imwrite("road_multibuffer.tif", dst6)
cv2.imshow('dst6',dst6)
cv2.waitKey(0)
cv2.destroyAllWindows()