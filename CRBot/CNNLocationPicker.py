# Import dependencies
import torch 
import numpy as np
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

def resizeImageCNN(x, h, w):
    newArray = np.zeros((1, h, w, 1))
    #newArray = newArray.reshape((-1, -1, 1))
    for column in x.T:
        img = Image.fromarray(column[:1900].reshape((38, 50)))
        img = img.resize((h,w), resample=Image.Resampling.BILINEAR)
        imgArray = np.asarray(img)
        imgArray = imgArray.reshape((1,h,w,1))
        newArray = np.append(newArray, imgArray, axis = 3)
        
    print(newArray.shape)
    return newArray[:,:,:,1:]


def resizeColouredImageCNN(x, h, w):
    newArray = np.zeros((3, h, w, 1))
    counter = 0
    #newArray = newArray.reshape((-1, -1, 1))
    for column in x.T:
        imgArray = column[:127500].reshape((250, 170, 3))
        #img = img.resize((250, 170, 3), resample=Image.Resampling.BILINEAR)
        imgArray = imgArray.reshape((3,h,w,1))
        newArray = np.append(newArray, imgArray, axis = 3)
        counter += 1
        print(counter)
    print(newArray.shape)
    return newArray[:,:,:,1:]


class LocationPicker(torch.nn.Module): 
    def __init__(self):
        super(LocationPicker, self).__init__()
        self.layernorm1 = nn.LayerNorm([3, 250, 170])
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1)
        self.layernorm2 = nn.LayerNorm([1, 32, 125, 85])
        #nn.ReLU(),
        self.pool = torch.nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.layernorm3 = nn.LayerNorm([1, 64, 62, 42])
        #nn.ReLU(),
        #torch.nn.MaxPool2d((2, 2)),
        self.fc1 = nn.Linear(64*62*42,256)
        self.layernorm4 = nn.LayerNorm([1, 256])
        self.fc2 = nn.Linear(297, 128)
        self.layernorm5 = nn.LayerNorm([1, 128])
        self.fc3 = nn.Linear(128, 2)
        

    def forward(self, x_im, x_var): 
        #x = self.layernorm1(x_im)
        x = torch.nn.functional.relu(self.pool(self.conv1(x_im)))
        #x = self.layernorm2(x)
        x = torch.nn.functional.relu(self.pool(self.conv2(x)))
        #x = self.layernorm3(x)
        #print("size = ",x.shape)
        x = x.view(-1, 64*62*42)
        x = torch.nn.functional.relu(self.fc1(x))
        #x = self.layernorm4(x)
        x = torch.cat((x, x_var), 1)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)
   

clf = LocationPicker().to('cuda')
opt = Adam(clf.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training flow 
if __name__ == "__main__": 
    losses = []


    a = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/170X250ColoredGame10Data.csv.npy", "r")
    b = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/170X250ColoredGame11Data.csv.npy", "r")
    c = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/170X250ColoredGame12Data.csv.npy", "r")
    d = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/170X250ColoredGame13Data.csv.npy", "r")


    data = np.hstack((a, b, c, d))
    #print(data)
    print(data.shape)

    indicies = []
    index = 0
    for column in data.T:
        if column[127541:].sum() == 0:
            indicies.append(index)
        index += 1

    data = np.delete(data, indicies, 1)
    print(data.shape)

    X_im = data[:127500, :]
    X_var = data[127500:127541, :]

    #X_im = resizeColouredImageCNN(X_im, 250, 170)
    #print("X_im Shape: ", np.shape(X_im))
    #np.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/X_im_L.csv", X_im, allow_pickle=False)
    #np.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/X_im_L", X_im, allow_pickle=False)
    
    X_im = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/X_im_L.csv.npy", "r")
    #X_im = resizeImageCNN(X_im, 250, 170)
    #print(X)
    print(X_im.shape)
    print(X_var.shape)

    Y = data[127541:, :]
    print(Y)

    X_im = np.transpose(X_im)
    X_var = np.transpose(X_var)
    Y = np.transpose(Y)

    tensor_x_im = torch.Tensor(X_im) # transform to torch tensor
    tensor_x_var = torch.Tensor(X_var) 
    tensor_y = torch.Tensor(Y)

    tensor_x_im = tensor_x_im.transpose(1,3)

    print(tensor_x_im.shape)
    print(tensor_x_var.shape)
    print(tensor_y.shape)

    my_dataset = TensorDataset(tensor_x_im, tensor_x_var, tensor_y) # create your datset


    train_size = int(0.8 * len(my_dataset))
    test_size = len(my_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset) # create your dataloader
    test_dataloader = DataLoader(test_dataset) # create your dataloader
    counter = 0
    for epoch in range(10): # train for 10 epochs
        for batch in train_dataloader: 
            #print("batch = ", batch)
            x_im, x_var, y = batch
            x_im, x_var, y = x_im.to("cuda"), x_var.to("cuda"), y.to("cuda")
            yhat = clf(x_im, x_var)
            y =  torch.squeeze(y)
            y[0] = (y[0]-1000)/900
            y[1] = (y[1]-200)/1300
            #print("y", y[0])
            if counter % 100 == 0:
                print("yhat=",yhat)
                print("y=",y)
            counter +=1
  
            loss = loss_fn(yhat, y)
            #print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"--------------------Epoch:{epoch} loss is {loss.item()}-----------------------")

    with open('CNN_location_picker_model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f) 



def predict(im, var):
    prm = LocationPicker().to('cuda')
    with open('CNN_location_picker_model_state.pt', 'rb') as f:
        prm.load_state_dict(load(f))
    prm.eval()
    tensorData_im = torch.tensor(im, dtype=torch.float32).to('cuda')
    tensorData_var = torch.tensor(var, dtype=torch.float32).to('cuda')
    return prm(tensorData_im, tensorData_var)
    



def predict_with_reinforcement_learning(im, var, y, reward):
    global var_saved, im_saved, yhatSaved, ySaved
    prm = LocationPicker().to('cpu')
    with open('CNN_location_picker_model_state.pt', 'rb') as f:
        prm.load_state_dict(load(f))

    tensorData_im = torch.tensor(im, dtype=torch.float32).to('cpu')
    tensorData_var = torch.tensor(var, dtype=torch.float32).to('cpu')

    yhat = prm(tensorData_im, tensorData_var)

    #if(len(var_saved) > 10):
        #var_saved.pop[0]
    #if(len(im_saved) > 10):
        #im_saved.pop[0]


    return yhat, tensorData_var, tensorData_im, y
   
def apply_reinforcement_learning(im_saved, var_saved, yhatSaved, ySaved, rewardSaved):
    prm = LocationPicker().to('cpu')
    with open('CNN_location_picker_model_state.pt', 'rb') as f:
        prm.load_state_dict(load(f))  
    opt = Adam(prm.parameters(), lr=0.01)
    loss_fn = nn.MSELoss() 
    for num in range(len(ySaved) - 3):
        reward = sum(rewardSaved[num: min(num+10, len(rewardSaved))])
        print("im", type(im_saved[num]))
        print("var", type(var_saved[num]))
        print("y", type(ySaved[num]))
        b = prm(im_saved[num], var_saved[num])
        print("learn: ", b)
        print("rewardPos", rewardSaved[num])
        loss = loss_fn(b, ySaved[num]) * reward/5000
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(num)
        with open('CNN_location_picker_model_state.pt', 'wb') as f: 
            save(clf.state_dict(), f) 