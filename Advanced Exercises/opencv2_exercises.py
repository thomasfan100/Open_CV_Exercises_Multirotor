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

if __name__ == "__main__":
    '''
    #MONTE CARLO SIMULATION
    NUM_TEST = 1000000 #multiple number of wins to get winrate
    rolls = craps(NUM_TEST)
    '''