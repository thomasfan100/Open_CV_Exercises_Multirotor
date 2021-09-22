#mahtoj polish matteo matt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter, convolve
import cv2

#PROBLEM 4
img = mpimg.imread('./strawberry.jpg')
img2 = mpimg.imread('./strawberry-solved.jpg')
img_height, img_width, _ = img.shape

img = img[np.arange(0,img_height):]
plt.imshow(img)
plt.show()
plt.imshow(img2) #x: 150 -> 600  y: 225 -> 450
plt.show()

'''
#PROBLEM 5
img = mpimg.imread('./rose-piano.jpg')
img = img / 255
blur_kernel = [[
    [1/9,1/9,1/9],
    [1/9,1/9,1/9],
    [1/9,1/9,1/9]]]

blur_kernel = np.array(blur_kernel)
blur_kernel /= np.sum(blur_kernel)  # ensure kernel sums to ~1

blurred = convolve(img, blur_kernel)
plt.imshow(blurred)
plt.show()


 # THIS IS FOR PART B
img2 = mpimg.imread('./rose-piano.jpg')
blur = cv2.GaussianBlur(img2,(5,5),0)
plt.imshow(blur)
plt.show()
'''
'''
#PROBLEM 3
img1 = mpimg.imread('./striped-leaf-pattern-0.jpg')
img2 = mpimg.imread('./striped-leaf-pattern-1.jpg')
binarize = np.where(img2>0,img2,img1)
plt.imsave("binarize.jpg",binarize)
#plt.imshow(binarize)
#plt.show()
'''
'''
#PROBLEM 2
img = mpimg.imread('./trump-putin.jpg')
#img = img[0:200,100:300]
img = img[200:400,400:600]
plt.imshow(img)
plt.show()
'''
'''
#PROBLEM 1
img = np.zeros(shape=(10,10))
#y, x
img[2:9,1:2] = 1
img[1:2,1:9] = 1
img[2:8,8:9] = 1
img[7:8,3:8] = 1
img[3:7,3:4] = 1
img[3:4,4:7] = 1
img[4:6,6:7] = 1
img[5:6,5:6] = 1
#img[6:7,2:8] = 1
plt.imshow(img,cmap='gray')
plt.show()
'''