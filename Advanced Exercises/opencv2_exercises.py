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
    
    img2 = np.where(img2 == [243,65,53],[255,247,0], img2)
    #hair
    img2 = np.where(img2 == [127,128,0], [0,0,0], img2)
    #skin
    img2 = np.where(img2 == [255,151,0],[205,151,1], img2)
    #background
    img2 = np.where(img2 == [254,254,254],[40,40,40], img2)
    #img3 = np.where(img3 == [254,254,254],[40,40,40], img3)
    img4 = np.where(img4 == [254,254,254],[40,40,40], img4)
    #overalls
    img3r = img3[:,:,0]
    img3g = img3[:,:,1]
    img3b = img3[:,:,2]

    a = np.array(
    [[1,2,3],
    [4,5,6],
    [7,8,3]])
    r = a[:,0]
    g = a[:,1]
    b = a[:,2]
    mask = np.where((r == 1) & (g == 2) & (b == 3), 1,0)
    mask = np.dstack((mask,mask,mask))
    a = np.where(mask,[4,5,6],a)
    print(a)
    
    #img3 = np.where(img3 == [243,65,53],[160,32,239],img3)
    #hand/skin
    img3 = np.where(img3 == (255,151,0) , [255,255,255],img3)
    #img3 = np.where(img3 == 254 ,[255,255,255],img3 )
    #shirt
    #img3 = np.where(img3 == [127,128,254],[255,247,0], img3)
    
    img = np.concatenate((img2,img3),axis = 0)
    img = np.concatenate((img,img4), axis = 0 )
    return img3

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
    '''
    plt.imshow(img)
    plt.show()
    plt.imshow(warioo)
    plt.show()
    '''
    '''
    #tomatoes
    img = mpimg.imread('./tomatoes.jpg')
    plt.imshow(img)
    plt.show()
    '''