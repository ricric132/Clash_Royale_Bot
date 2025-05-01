import numpy as np
from PIL import Image, ImageGrab, ImageChops, ImageOps
import threading
import torch 
import time

def printNP(path):
    a = np.load(path, "r")
    print(a)
    print(a.shape)
    print(a[1941:,:])

#def saveSnip(path, Im, dimensions, grayscale):
"""
im2 = ImageGrab.grab(bbox = (1050, 200, 1900, 1450))
im2 = ImageOps.grayscale(im2)
im2 = im2.resize((38, 50), resample=Image.Resampling.BILINEAR)
im2.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/BaseMap38X50Gray.jpg")
baseMap = Image.open("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/BaseMap38X50Gray.jpg")
"""

#printNP("game8Data.csv.npy")
#a = np.array([0, 0, 0, 0])
#print(np.where(a == 1)[0])
'''
a = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/170X250ColoredGame10Data.csv.npy", "r")
print(np.shape(a[:, 0]))

b = a[:127500, 33]
print(np.shape(b))

c = Image.fromarray((np.reshape(b, (250, 170, 3))).astype(np.uint8))

c.show()

X_im = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/X_im.csv.npy", "r")
print(np.shape(X_im))
'''

'''
cardPreds = torch.rand(1, 9)
card1 = np.zeros((8, 1))
card1[5][0] = 1
print(cardPreds)
print(card1)
print(np.where(card1 == 1)[0])
print(cardPreds[0][np.where(card1 == 1)[0]])
'''

'''
time.sleep(1)

nohp = np.array([32,32,32])

princesshp1 = ImageGrab.grab(bbox = (1156, 349, 1265, 350))
princesshp2 = ImageGrab.grab(bbox = (1697, 349, 1806, 350))
princesshp3 = ImageGrab.grab(bbox = (1697, 1196, 1806, 1210))
princesshp3 = np.rot90(princesshp3,3)
princesshp4 = ImageGrab.grab(bbox = (1156, 1196, 1265, 1210))
princesshp4 = np.rot90(princesshp4,3)
#princesshp4 = ImageGrab.grab(bbox = (1697, 349, 1806, 350))
#princesshp3 = Image.fromarray(princesshp3)
#princesshp3.show()
princesshp4 = Image.fromarray(princesshp4)
princesshp4.show()
#np.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/princesshp1", np.asarray(princesshp1))
princesshp1 = np.asarray(princesshp1) - np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/princesshp1.npy", "r")
princesshp1 = np.flip(princesshp1, axis=1)
princesshp2 = np.asarray(princesshp2) - np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/princesshp2.npy", "r")
princesshp2 = np.flip(princesshp2, axis=1)

#princesshp3 = np.asarray(princesshp3)
princesshp3 = np.flip(princesshp3, axis=1)
princesshp4 = np.flip(princesshp4, axis=1)

hp1 = 3052
hp2 = 3052
hp3 = 0
hp4 = 0

for x in princesshp1:
    for y in x:
        if np.sum(y) > 100:
            hp1 = hp1 - 3052/109
        else:
            break

for x in princesshp2:
    for y in x:
        if np.sum(y) > 100:
            hp2 = hp2 - 3052/109
        else:
            break

findGrey = False

for x in princesshp3:
    if findGrey == True:
        break
    hp3 += 3052/109
    for y in x:
        if y[0] > 70 and y[0] < 73 and y[1] > 86 and y[1] < 90 and y[2] > 119 and y[2] < 128:
            print(y)
            findGrey = True

findGrey = False

for x in princesshp4:
    if findGrey == True:
        break
    hp4 += 3052/109
    for y in x:
        if y[0] > 70 and y[0] < 73 and y[1] > 86 and y[1] < 90 and y[2] > 119 and y[2] < 128:
            print(y)
            findGrey = True
    
            

#print(princesshp3.shape)
#print(princesshp3[108: , :, :])
print(hp1)
print(hp2)
print(hp3)
print(hp4)
'''
'''
time.sleep(1)

a = ImageGrab.grab(bbox=(1405, 1450, 1570, 1451))
a.show()
print(np.asarray(a)[0][0])

hp2 = 4824
for x in np.rot90(np.asarray(a), 2):
    for y in x:
        if y[0] != 105 or y[1] != 198 or y[2] != 241:
            hp2 = hp2 - 4824/165
        else:
            break
'''

a = torch.tensor([2, 1, 4])
b = torch.zeros((1, 3))
b[0][torch.argmax(a)] = 1
print(b)

