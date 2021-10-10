#car
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter, convolve
import cv2
import random

def craps(rolls):
    #roll 1
    tests = roll(rolls)
    wins = np.where((tests == 7) | (tests ==11))
    tests = np.delete(tests,wins)
    numWins = np.size(wins)
    losses = np.where((tests == 2)|(tests == 3)|(tests == 12))
    tests = np.delete(tests,losses)
    numLosses = np.size(losses)

    #roll2
    while(np.size(tests) != 0):
        tests2 = roll(np.size(tests))

        wins = np.where(tests == tests2)
        numWins += np.size(wins)
        tests = np.delete(tests,wins)
        tests2 = np.delete(tests2,wins)

        losses = np.where(tests2 == 7)
        numLosses += np.size(losses)
        tests = np.delete(tests, losses)
    
    print("Win Ratio:",(numWins/ rolls))
    print("Loss Ratio:", (numLosses/rolls))

def roll(rolls):
    return np.random.randint(1,7,size=rolls) + np.random.randint(1,7,size=rolls)

def wario(img):
    #split image
    img2 = img[0:210]
    img3 = img[210:400]
    img4 = img[400:]
    #hat
    img2 = changeColor(img2,[243,65,53],[255,247,0])
    #hair
    img2 = changeColor(img2,[127,128,0], [0,0,0])
    #skin
    img2 = changeColor(img2,[255,151,0],[205,151,1])
    #background
    img2 = np.where(img2 == [254,254,254],[40,40,40], img2)
    img3 = changeColor(img3,[254,254,254],[40,40,40])
    img4 = np.where(img4 == [254,254,254],[40,40,40], img4)
    #overalls
    img3 = changeColor(img3,[243,65,53],[160,32,239])
    #hand/skin
    img3 = changeColor(img3,[255,151,0] , [255,255,255])
    #shirt
    img3 = changeColor(img3,[127,128,0],[255,247,0])

    img = np.concatenate((img2,img3),axis = 0)
    img = np.concatenate((img,img4), axis = 0 )
    img = img[::2]
    return img

def changeColor(img,color, newcolor):
    '''
    Helper function for mario program. Changes the color by splitting into three channels. 
    Returns an image with colors converted.
    '''
    imgr = img[:,:,0]
    imgg = img[:,:,1]
    imgb = img[:,:,2]
    #210 235 250= 235
    #mask = np.where((abs(imgr - color[0]) < 100) & (abs(imgg - color[1]) < 100) & (abs(imgb - color[2]) < 100), 1,0)
    mask = np.where((imgr == color[0]) & (imgg == color[1]) & (imgb == color[2]), 1,0)
    mask = np.dstack((mask,mask,mask))
    img = np.where(mask, newcolor, img)
    return img

def tomato(image):
    vectorized = img.reshape((-1, 3)) #makes sure its 3 columns?
    #idxs = np.array([idx for idx, _ in np.ndenumerate(np.mean(img, axis=2))]) #gets all indexes, [0,1],[0,2],etc.
    #vectorized = np.hstack((vectorized,idxs)) #adds x,y indexes to vectorized
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 9 #Number of colors we want
    _, label, center = cv2.kmeans(np.float32(vectorized), K, bestLabels=None, criteria=termination_criteria, attempts=10, flags=0)

    k_image = np.uint8(center)[label.flatten()]
    k_image = k_image.reshape((image.shape))
    
    
    ## Mask for reds
    img_hsv = cv2.cvtColor(k_image, cv2.COLOR_RGB2HSV)
    plt.imshow(img_hsv)
    plt.show()
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    mask = np.dstack((mask, mask, mask))

    output_image = np.copy(image)
    output_image = np.where(mask==(0, 0, 0), output_image, 255 - output_image)
    return output_image

if __name__ == "__main__":
    '''
    #MONTE CARLO SIMULATION
    NUM_TEST = 1000000 #multiple number of wins to get winrate
    rolls = craps(NUM_TEST)
    '''
    '''
    #Mario Bros
    wario_solution = mpimg.imread('./wario.png')
    img = mpimg.imread('./mario.jpg')
    img = wario(img)
    
    plt.imshow(img)
    plt.show()
    plt.imshow(wario_solution)
    plt.show()
    '''
    
    #tomatoes
    img = mpimg.imread('./tomatoes.jpg')
    img = tomato(img)
    plt.imshow(img)
    plt.show()
    