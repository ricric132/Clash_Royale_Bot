import torch 
from torch import nn, save, load
from torch.optim import Adam
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot
from PIL import Image
from torchvision.transforms import ToTensor

def resizeImage(x, h, w):
    newArray = np.zeros((h*w+33, 1))
    newArray = newArray.reshape(-1, 1)
    for column in x.T:
        img = Image.fromarray(column[:1900].reshape((38, 50)))
        img = img.resize((h,w), resample=Image.Resampling.BILINEAR)
        imgArray = np.asarray(img)
        imgArray = imgArray.flatten()
        otherData = column[1900:]
        temp = np.append(imgArray, otherData)
        temp = np.transpose(temp)
        temp = temp.reshape(-1, 1)
        newArray = np.append(newArray, temp, axis=1)
    return newArray[:, 1:]

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(508, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.model(x)



prm = net().to("cuda")
opt = Adam(prm.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
counter = 0

if __name__ == "__main__":
    losses = []


    a = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/game1Data.csv.npy", "r")
    b = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/game2Data.csv.npy", "r")
    c = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/game3Data.csv.npy", "r")
    d = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/game4Data.csv.npy", "r")
    e = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/game5Data.csv.npy", "r")
    f = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/game6Data.csv.npy", "r")
    g = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/game7Data.csv.npy", "r")
    h = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/game8Data.csv.npy", "r")

    data = np.hstack((a, b, c, d, e, f, g, h))
    #print(data)
    #print(data.shape)



    X = data[:1933, :]



    X = resizeImage(X, 19, 25)
    #print(X)
    #print(X.shape)

    Y = data[1933:1941, :]
    noVal = np.zeros(Y.shape[1])
    noVal = noVal.reshape(-1, 1)


    print(Y.shape)
    counter = 0
    for column in Y.T:
        if np.sum(column) == 1:
            for num in range(len(column)):
                if column[num] == 1:
                    if np.sum(Y.T[counter - 1][:]) == 0:
                        Y.T[counter - 1][num] = 1
                    if counter + 1 >= Y.shape[1]:
                        print("end")
                        break
                    if np.sum(Y.T[counter + 1][:]) == 0:
                        Y.T[counter + 1][num] = 1
                    break
        counter += 1

    counter = 0
    for column in Y.T:
        if np.sum(column) == 0:
            noVal[counter] = 1
        counter += 1

    Y = np.append(Y, noVal.T, axis=0)
    np.savetxt('C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/SampleData.csv', Y, delimiter=',')
    #print(Y)
    #print(Y.shape)

    X = np.transpose(X)
    Y = np.transpose(Y)

    tensor_x = torch.Tensor(X) # transform to torch tensor
    tensor_y = torch.Tensor(Y)

    #print(tensor_x.shape)
    #print(tensor_y.shape)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset, 1) # create your dataloader

    for epoch in range(10):
        for batch in my_dataset:
            x, y = batch
            x, y = x.to("cuda"), y.to("cuda")
            yhat = prm(x)
            if counter % 500 == 0:
                print(yhat)
            counter +=1
            loss = loss_fn(yhat, y)
            #loss = loss * 10
            print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        #losses.append(loss.item())
        print(f"--------------------Epoch:{epoch} loss is {loss.item()}-----------------------")

    print("done")
    
    with open('DNN_model_state.pt', 'wb') as f: 
        save(prm.state_dict(), f) 

    #with open('DNN_model_state.pt', 'rb') as f:
        #prm.load_state_dict(load(f))

    #print(load('DNN_model_state.pt'))
    prm.eval()
    dataTensor = torch.rand(1, 508).to('cuda')
    print(prm(dataTensor))

    dataTensor = torch.rand(1, 508).to('cuda')
    print(prm(dataTensor))


#pyplot.plot(losses)
#pyplot.show()

def predict(data):
    with open('DNN_model_state.pt', 'rb') as f:
        prm.load_state_dict(load(f))
    prm.eval()
    tensorData = torch.tensor(data, dtype=torch.float32).to('cuda')
    return prm(tensorData)
