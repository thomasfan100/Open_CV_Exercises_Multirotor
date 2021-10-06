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
    img3 = img[210:]
    
    img2 = np.where(img2 == [243,65,53],[255,247,0], img2)
    img2 = np.where(img2 == [127,128,0], [0,0,0], img2)
    img2 = np.where(img2 == [255,151,0],[205,151,1], img2)
    img = np.concatenate((img2,img3),axis = 0)
    #img3 = img[210:]
    #reds = np.where(meanb > 250,255,25)

    return img

if __name__ == "__main__":
    '''
    #MONTE CARLO SIMULATION
    NUM_TEST = 1000000 #multiple number of wins to get winrate
    rolls = craps(NUM_TEST)
    '''
    #Mario Bros
    warioo = mpimg.imread('./wario.png')
    img = mpimg.imread('./mario.jpg')
    img = wario(img)
    
    plt.imshow(img)
    plt.show()
    plt.imshow(warioo)
    plt.show()