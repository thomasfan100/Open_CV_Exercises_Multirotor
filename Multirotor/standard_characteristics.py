import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter, convolve
import numpy as np
import cv2 

class Standard_Characteristics:
    def __init__(self, col, shp) -> None:
        self.color = col
        self.shape = shp

    def printResults(self) -> None:
        print(f"The color is {self.color} and the shape is {self.shape}")

def nothing(x):
    pass
def hsvslider(img):
    
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
    while True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        
        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(img,img, mask= mask)
        cv2.imshow('res',res)
        k = cv2.waitKey(5) 
        if k == 27:
            break
        #cv2.imshow('image', img)
        #cv2.waitKey(0)
def findCountours(img):
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 1)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)

    # Find contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and filter for ROI
    image_number = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = original[y:y+h, x:x+w]
        cv2.imwrite("./imagesgenerated/ROI_{}.png".format(image_number), ROI)
        image_number += 1

    cv2.imshow('canny', canny)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = img.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    plt.imshow(image_copy)
    plt.show()
    '''
if __name__ == '__main__':
    #img = cv2.imread('./images/basic.jpg')
    img = cv2.imread('./images/basic2.jpg')
    #img = cv2.imread("./images/SUAS2012.jpg")
    #findCountours(img)
    hsvslider(img)
    

 