import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

#READ ALL IMAGES
img1 = cv2.imread('/Users/utkupolat/Desktop/COMPE464-Homework2/images_for_HW2/im1.bmp',0)
img2 = cv2.imread('/Users/utkupolat/Desktop/COMPE464-Homework2/images_for_HW2/im2.bmp',0)
img3 = cv2.imread('/Users/utkupolat/Desktop/COMPE464-Homework2/images_for_HW2/im3.bmp',0)
img4 = cv2.imread('/Users/utkupolat/Desktop/COMPE464-Homework2/images_for_HW2/im4.jpg',0)
################################


#HISTOGRAMS OF ALL INPUT IMAGES
plt.hist(img1.ravel(),256,[0,256])
#plt.show()

plt.hist(img2.ravel(),256,[0,256])
#plt.show()

plt.hist(img3.ravel(),256,[0,256])
#plt.show()

plt.hist(img4.ravel(),256,[0,256])
#plt.show()
################################


#APPLY THE HISTOGRAM EQUALIZATION ALL IMAGES AND SHOW THEIR HISTOGRAMS
img1HistEgu = cv2.equalizeHist(img1)
HistogramImage_1 = np.hstack((img1,img1HistEgu)) 
#cv2.imwrite('HistogramImage_1.png',HistogramImage_1)

plt.hist(HistogramImage_1.ravel(),256,[0,256])
#plt.show()

img2HistEgu = cv2.equalizeHist(img2)
HistogramImage_2 = np.hstack((img2,img2HistEgu)) 
#cv2.imwrite('HistogramImage_2.png',HistogramImage_2)

plt.hist(HistogramImage_2.ravel(),256,[0,256])
#plt.show()

img3HistEgu = cv2.equalizeHist(img3)
HistogramImage_3 = np.hstack((img3,img3HistEgu)) 
#cv2.imwrite('HistogramImage_3.png',HistogramImage_3)

plt.hist(HistogramImage_3.ravel(),256,[0,256])
#plt.show()

img4HistEgu = cv2.equalizeHist(img4)
HistogramImage_4 = np.hstack((img4,img4HistEgu)) 
#cv2.imwrite('HistogramImage_4.png',HistogramImage_4)

plt.hist(HistogramImage_4.ravel(),256,[0,256])
#plt.show()
#################################



#APPLY THE LOCAL HISTOGRAM EQUALIZATION OF ALL IMAGES AND SHOW THEIR HISTOGRAMS
localImage1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
localHist1 = localImage1.apply(img1)
#cv2.imwrite('LocalImage_1.jpg',localHist1)

plt.hist(localHist1.ravel(),256,[0,256])
plt.show()

localImage2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
localHist2 = localImage2.apply(img2)
#cv2.imwrite('LocalImage_2.jpg',localHist2)

plt.hist(localHist2.ravel(),256,[0,256])
plt.show()

localImage3 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
localHist3 = localImage3.apply(img3)
#cv2.imwrite('LocalImage_3.jpg',localHist3)

plt.hist(localHist3.ravel(),256,[0,256])
plt.show()

localImage4 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
localHist4 = localImage4.apply(img4)
#cv2.imwrite('LocalImage_4.jpg',localHist4)

plt.hist(localHist4.ravel(),256,[0,256])
plt.show()
#################################






