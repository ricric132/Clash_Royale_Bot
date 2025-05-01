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
    #im2 = ImageOps.grayscale(im2)850 1250
    im2 = im2.resize((170, 250), resample=Image.Resampling.BILINEAR)
    #toSave = im2
    #toSave = toSave.resize((170, 250), resample=Image.Resampling.BILINEAR)
    #toSave.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/BaseMap170X250.jpg")
    baseMap = Image.open("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/BaseMap170X250.jpg")

    # finding difference
    diff = ImageChops.difference(baseMap, im2)

    return np.asarray(diff)



#card1.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/Card1.jpg")
#card2.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/Card2.jpg")
#card3.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/Card3.jpg")
#card4.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/Card4.jpg")

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





playing = True
elixer = get_elixer()
card1 = np.zeros((8, 1))
card2 = np.zeros((8, 1))
card3 = np.zeros((8, 1))
card4 = np.zeros((8, 1))
screen = get_map().flatten()
selected_card = np.zeros((8, 1))
clickX = 0
clickY = 0
data = np.zeros(screen.size + 43)
data = data.reshape(-1, 1)
print(data.shape)

def find_variables():
    global card1, card2, card3, card4, data
    while playing == True:
        screen = get_map().flatten().reshape(-1, 1)
        get_cards()
        elixer = get_elixer()
        frameData = np.vstack((screen, card1, card2, card3, card4, elixer, selected_card, clickX, clickY))
        data = np.append(data, frameData, axis=1)  
        print(playing)
    print(data)
    print(data.shape)
    np.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/170X250ColoredGame13Data.csv", data, allow_pickle=False)
    


        
#find_variables()

def track_clicks():
    Xmax = 1900 
    Xmin = 1000
    Ymin = 200
    Ymax = 1500
    def on_click(x, y, button, pressed):
        global playing, clickX, clickY, selected_card
        if pressed == False:
            if str(button) == "Button.left":
                clickX = 0
                clickY = 0
                if(y > 1590):
                    if x > start and x < start + cardWidth:
                        selected_card = card1
                    elif x > start + cardWidth + gap and x < start + (2*cardWidth) + gap:
                        selected_card = card2
                    elif x > start + (2 * cardWidth) + (2 * gap) and x < start + (3 * cardWidth) + (2 * gap):
                        selected_card = card3
                    elif x > start + (3 * cardWidth) + (3 * gap) and x < start + (4 * cardWidth) + (3 * gap):
                        selected_card = card4
                else:
                    clickX = min(Xmax, max((x, Xmin)))
                    clickY = min(Ymax, max((y, Ymin)))
                    selected_card = np.zeros((8, 1))

            if str(button) == "Button.right":
                playing = False


    with Listener(on_click=on_click) as listener:
        listener.join()




# creating thread
t1 = threading.Thread(target=find_variables)
t2 = threading.Thread(target=track_clicks)



 
# starting thread 1
t1.start()
# starting thread 2
t2.start()


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


