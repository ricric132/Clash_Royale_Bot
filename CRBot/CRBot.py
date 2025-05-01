from audioop import minmax
from decimal import Clamped
from operator import truediv
from turtle import onclick
from PIL import Image, ImageGrab, ImageChops, ImageOps
import time
import numpy as np
import pyautogui
from pynput.mouse import Listener
import threading
import DNN
import CNN
import CNNLocationPicker 
import torch

time.sleep(1)


def one_hot(num):
    oneHotCard = np.zeros((8, 1))
    oneHotCard[num] = 1
    return oneHotCard

cards = ["knights", "minion", "arrow", "archer", "fireball", "musketeer", "minipeka", "giant", "undefined"]
cardColorValue1 = [(138, 121, 118), (59, 86, 129), (147, 158, 194), (117, 91, 115), (170, 124, 82), (118, 107, 142), (78, 102, 143), (146, 100, 87)] 
cardColorValue2 = [(138, 121, 118), (59, 86, 129), (147, 158, 194), (117, 91, 115), (170, 124, 82), (118, 107, 142), (78, 102, 143), (146, 100, 87)]
cardColorValue3 = [(138, 121, 118), (59, 87, 130), (147, 158, 194), (117, 92, 115), (170, 124, 82), (118, 107, 142), (78, 102, 143), (146, 100, 87)]
cardColorValue4 = [(139, 121, 119), (59, 87, 130), (147, 158, 194), (117, 92, 115), (170, 124, 82), (118, 107, 142), (78, 102, 143), (146, 100, 87)]


cardWidth = 185
gap = 10
start = 1180

def get_cards():
    global card1, card2, card3, card4

    card1pic = ImageGrab.grab(bbox = (start, 1590, start + cardWidth, 1810))
    card2pic = ImageGrab.grab(bbox = (start + cardWidth + gap, 1590, start + (2*cardWidth) + gap, 1810))
    card3pic = ImageGrab.grab(bbox = (start + (2 * cardWidth) + (2 * gap), 1590, start + (3 * cardWidth) + (2 * gap), 1810))
    card4pic = ImageGrab.grab(bbox = (start + (3 * cardWidth) + (3 * gap), 1590, start + (4 * cardWidth) + (3 * gap), 1810))

    card1Small = card1pic.resize((1,1), resample=Image.Resampling.BILINEAR)
    card2Small = card2pic.resize((1,1), resample=Image.Resampling.BILINEAR)
    card3Small = card3pic.resize((1,1), resample=Image.Resampling.BILINEAR)
    card4Small = card4pic.resize((1,1), resample=Image.Resampling.BILINEAR)

    card1pix = card1Small.getpixel((0, 0))
    if card1pix in cardColorValue1:
        card1 = one_hot(cardColorValue1.index(card1pix))


    card2pix = card2Small.getpixel((0, 0))
    if card2pix in cardColorValue2:
        card2 = one_hot(cardColorValue2.index(card2pix))


    card3pix = card3Small.getpixel((0, 0))
    if card3pix in cardColorValue3:
        card3 = one_hot(cardColorValue3.index(card3pix))


    card4pix = card4Small.getpixel((0, 0))
    if card4pix in cardColorValue4:
        card4 = one_hot(cardColorValue4.index(card4pix))
        
    return card1, card2, card3, card4

def get_map():
    im2 = ImageGrab.grab(bbox = (1050, 200, 1900, 1450))
    #im2 = ImageOps.grayscale(im2)
    im2 = im2.resize((190,250), resample=Image.Resampling.BILINEAR)
    #toSave = ImageOps.grayscale(im2)
    #toSave = toSave.resize((38,50), resample=Image.Resampling.BILINEAR)
    #toSave.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/BaseMap.jpg")
    
    baseMap = Image.open("BaseMap170X250.jpg")
    # finding difference
    diff = ImageChops.difference(baseMap, im2)
    #print("map = ", np.asarray(diff))
    return np.asarray(diff)

def get_elixer():
    elixerBarPixels = [1942, 1880, 1810, 1736, 1668, 1594, 1522, 1450, 1378, 1306]

    elixerBar = ImageGrab.grab(bbox = (1100, 1820, 2000, 1880))
    tempElixer = 0 
    for pixel in elixerBarPixels:
        pixelValue = elixerBar.getpixel((pixel - 1100, 30))
        if pixelValue[0] > 10:
            tempElixer = 10 - elixerBarPixels.index(pixel)
            break
    
    return tempElixer

def get_hp():
    princesshp1 = ImageGrab.grab(bbox = (1156, 349, 1265, 350))
    princesshp2 = ImageGrab.grab(bbox = (1697, 349, 1806, 350))
    princesshp3 = ImageGrab.grab(bbox = (1697, 1196, 1806, 1210))
    princesshp3 = np.rot90(princesshp3,3)
    princesshp4 = ImageGrab.grab(bbox = (1156, 1196, 1265, 1210))
    princesshp4 = np.rot90(princesshp4,3)
    khp1 = ImageGrab.grab(bbox=(1405, 1450, 1570, 1451))
    khp1 = np.asarray(khp1)
    #princesshp4 = ImageGrab.grab(bbox = (1697, 349, 1806, 350))
    #princesshp3 = Image.fromarray(princesshp3)
    #princesshp3.show()
    princesshp4 = Image.fromarray(princesshp4)
    #np.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/princesshp1", np.asarray(princesshp1))
    princesshp1 = np.asarray(princesshp1) - np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/princesshp1.npy", "r")
    princesshp1 = np.flip(princesshp1, axis=1)
    princesshp2 = np.asarray(princesshp2) - np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/princesshp2.npy", "r")
    princesshp2 = np.flip(princesshp2, axis=1)

    #princesshp3 = np.asarray(princesshp3)
    princesshp3 = np.flip(princesshp3, axis=1)
    princesshp4 = np.flip(princesshp4, axis=1)

    p3Active = False
    p4Active = False

    hp1 = 3052
    hp2 = 3052
    hp3 = 0
    hp4 = 0
    hpk1 = 4824
    for x in princesshp1:
        for y in x:
            if np.sum(y) > 100:
                hp1 = hp1 - 3052/110
            else:
                break

    for x in princesshp2:
        for y in x:
            if np.sum(y) > 100:
                hp2 = hp2 - 3052/110
            else:
                break

    findGrey = False

    for x in princesshp3:
        if findGrey == True:
            p3Active = True
            break
        hp3 += 3052/109
        for y in x:
            if y[0] > 70 and y[0] < 73 and y[1] > 86 and y[1] < 90 and y[2] > 119 and y[2] < 128:
                #print(y)
                findGrey = True
    if p3Active == False:
        hp3 = 3052

    findGrey = False

    for x in princesshp4:
        if findGrey == True:
            p4Active = True
            break
        hp4 += 3052/109
        for y in x:
            if y[0] > 70 and y[0] < 73 and y[1] > 86 and y[1] < 90 and y[2] > 119 and y[2] < 128:
                #print(y)
                findGrey = True
    if p4Active == False:
        hp3 = 3052
    


    for x in np.rot90(khp1, 2):
        for y in x:
            if y[0] != 105 or y[1] != 198 or y[2] != 241:
                hpk1 = hpk1 - 4824/165
                
            else:
                break
    


    return hp1, hp2, hp3, hp4, hpk1

def get_reward(): 
    global savedHP1, savedHP2, savedHP3, savedHP4, savedHPk1, savedElixer
    hp1, hp2, hp3, hp4, hpk1 = get_hp()
    elixer = get_elixer()
    reward = 0

    if(savedHP1 > 0):
        hpChange1 = savedHP1 - hp1
        savedHP1 = hp1
        if(hp1 <= 0):
            reward += 1000
        reward += hpChange1 * (1 + (3052 - hp1)/3052)


    if(savedHP2 > 0):
        hpChange2 = savedHP2 - hp2
        savedHP2 = hp2
        if(hp2 <= 0):
            reward += 1000
        reward += hpChange2 * (1 + (3052 - hp2)/3052)

    if(savedHP3 > 0):
        hpChange3 = savedHP3 - hp3
        savedHP3 = hp3
        if(hp3 <= 0):
            reward -= 1000
        reward -= hpChange3 * (1 + (3052 - hp3)/3052)

    if(savedHP4 > 0):
        hpChange4 = savedHP4 - hp4
        savedHP4 = hp4
        if(hp4 <= 0):
            reward -= 1000
        reward -= hpChange4 * (1 + (3052 - hp4)/3052)

    if(savedHPk1 > 0):
        hpChange2 = savedHPk1 - hpk1
        savedHPk1 = hpk1
        if(hp2 <= 0):
            reward -= 5000
        reward -= hpChange2 * (1 + (4824 - hp2)/4824)
    
    elixerChange = savedElixer - elixer
    savedElixer = elixer

    reward -=  elixerChange * 200


    if min(hp1, hp2) < min(hp3, hp4):
        reward += 100
    else:
        reward -= 10
    
    return reward
    





playing = True
elixer = get_elixer()
card1 = np.zeros((8, 1))
card2 = np.zeros((8, 1))
card3 = np.zeros((8, 1))
card4 = np.zeros((8, 1))
screen = get_map()
selected_card = np.zeros((8, 1))
clickX = 0
clickY = 0
data = np.zeros(screen.size + 43)
data = data.reshape(-1, 1)
#print(data.shape)
savedElixer = 5
savedHP1 = 3052
savedHP2 = 3052
savedHP3 = 3052
savedHP4 = 3052
savedHPk1 = 4824
savedHPk2 = 4824

var_saved1 = []
var_saved2 = []

im_saved1 = []
im_saved2 = []

yhatSaved1 = []
yhatSaved2 = []

ySaved1 = []
ySaved2 = []

rewardSaved = []


#creates a function called make_preds which has no inpute
def make_pred():
    #allows the funtion to access variables defined outside of the function
    global card1, card2, card3, card4, data, selected_card, playing
    savedPosition = None

    #runs the code indented repeatedly while the variable playing is true, so while the game is playing runs bellow code
    while playing == True:

        #checks if a particular pixel is a certain value that is the same as end screen to end end playing
        endCheck =  np.asarray(ImageGrab.grab(bbox = (1350, 1709, 1351, 1710)))[0,0] 
        if endCheck[0] == 34 and endCheck[1] == 152 and endCheck[2] == 255:
            playing = False
            break

        #uses custom functions to get data required and reformats it
        reward = get_reward()
        screen = get_map()
        screen = screen.reshape((1, 3, 250, 170))
        get_cards()
        elixer = get_elixer()
        frameData = np.vstack((card1, card2, card3, card4, elixer))
        frameData = np.transpose(frameData)

        #passes the data into the convolutional neural network, returning the data and prediction
        cardPreds, tensorData_var, tensorData_im, yhat, selected_card_return = CNN.predict_with_reinforcement_learning(screen, frameData, selected_card, reward)
        #cardPreds = CNN.predict(screen, frameData)
        b = torch.zeros((1, 9))
        b[0][torch.argmax(cardPreds)] = 1

        #saves data into lists for running it through reward system
        var_saved1.append(tensorData_var)
        im_saved1.append(tensorData_im)
        yhatSaved1.append(yhat)
        ySaved1.append(b)
        rewardSaved.append(reward)
        

        total = 0
        bestCard = np.zeros((8, 1))
        bestScore = 0
        cardPos = 0
        
        if card1.sum() > 0:
            #print('1.', cardPreds[np.where(card1 == 1)[0]])
            total += cardPreds[np.where(card1 == 1)[0]]
            if cardPreds[np.where(card1 == 1)[0]] > bestScore:
                bestScore = cardPreds[np.where(card1 == 1)[0]]
                bestCard = card1
                cardPos = 1300

        if card2.sum() > 0:
            total += cardPreds[np.where(card2 == 1)[0]]
            #print('2.', cardPreds[np.where(card2 == 1)[0]])
            if cardPreds[np.where(card2 == 1)[0]] > bestScore:
                bestScore = cardPreds[np.where(card2 == 1)[0]]
                bestCard = card2
                cardPos = 1500

        if card3.sum() > 0:
            total += cardPreds[np.where(card3 == 1)[0]]
            #print('3.', cardPreds[np.where(card3 == 1)[0]])
            if cardPreds[np.where(card3 == 1)[0]] > bestScore:
                bestScore = cardPreds[np.where(card3 == 1)[0]]
                bestCard = card3
                cardPos = 1700

        if card4.sum() > 0:
            total += cardPreds[np.where(card4 == 1)[0]]
            #print('4.', cardPreds[np.where(card4 == 1)[0]])
            if cardPreds[np.where(card4 == 1)[0]] > bestScore:
                bestScore = cardPreds[np.where(card4 == 1)[0]]
                bestCard = card4
                cardPos = 1800
        endCheck =  np.asarray(ImageGrab.grab(bbox = (1350, 1709, 1351, 1710)))[0,0] 
        if endCheck[0] == 34 and endCheck[1] == 152 and endCheck[2] == 255:
            playing = False
            break
            #print("selected:", selected_card)
        

        if bestScore > cardPreds[8]:
            pyautogui.click(cardPos, 1700)
            selected_card = bestCard


        
        if selected_card.sum() > 0:
            Xmax = 1850
            Xmin = 1050
            Ymin = 250
            Ymax = 1450

            position = CNNLocationPicker.predict(screen, np.transpose(np.vstack((card1, card2, card3, card4, elixer, selected_card))))

            #position, tensorData_var, tensorData_im, seleted_location = CNNLocationPicker.predict_with_reinforcement_learning(screen, np.transpose(np.vstack((card1, card2, card3, card4, elixer, selected_card))), savedPosition, reward)
            
            #var_saved2.append(tensorData_var)
            #im_saved2.append(tensorData_im)
            #yhatSaved2.append(position)
            #ySaved2.append(position)
            #rewardSaved.append(reward)
            position[0] = position[0] * 900 + 1000 
            position[1] = position[1] * 1300 + 200
            #print(position)
            pyautogui.click(min(Xmax, max((position[0], Xmin))), min(Ymax, max((position[1], Ymin))))
            selected_card = np.zeros((8, 1))
    #print("learning")
    #CNN.apply_reinforcement_learning(im_saved1, var_saved1, yhatSaved1, ySaved1, rewardSaved)
    #CNNLocationPicker.apply_reinforcement_learning(im_saved2, var_saved2, yhatSaved2, ySaved2, rewardSaved)




make_pred()



 

#cardsFinal = cardsSmall.resize(cards.size, Image.NEAREST)
#cardsFinal.show()
#print(cards.getpixel(()))
""""
result = np.array(result)
result = np.append(result, card1)
result = np.append(result, card2)
result = np.append(result, card3)
result = np.append(result, card4)
result = np.append(result, elixer)
result = np.transpose(result)
print("result = ", result)
print(result.shape)
result = result.reshape(-1, 1)
result = np.append(result, result, axis=1)
print(result)
print(result.shape)
"""


