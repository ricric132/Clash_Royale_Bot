# Import dependencies
import torch 
import numpy as np
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import random

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

# Get data 


# Image Classifier Neural Network
class ImageClassifier(torch.nn.Module): 
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.layernorm1 = nn.LayerNorm([3, 250, 170])
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1)
        self.layernorm2 = nn.LayerNorm([1, 32, 125, 85])
        #nn.ReLU(),
        self.pool = torch.nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.layernorm3 = nn.LayerNorm([1, 64, 62, 42])
        #nn.ReLU(),
        self.fc1 = nn.Linear(64*62*42,256)
        self.layernorm4 = nn.LayerNorm([1, 256])
        self.fc2 = nn.Linear(289, 128)
        self.layernorm5 = nn.LayerNorm([1, 128])
        self.fc3 = nn.Linear(128, 9)
        

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
        #x = self.layernorm5(x)
        x = self.fc3(x)
        return x.view(-1)

def test(test_loader, test_split): 
    # Load the model that we saved at the end of the training loop 
    model = ImageClassifier() 
    path = "CNN_model_state.pt" 
    model.load_state_dict(torch.load(path)) 
    model.eval()
    running_accuracy = 0 
    total = 0 
 
    with torch.no_grad(): 
        for data in test_loader: 
            inputs_im, inputs_var, outputs = data 
            outputs = outputs.to(torch.float32) 
            predicted_outputs = model(inputs_im, inputs_var) 
            predicted = torch.argmax(predicted_outputs) 
            total += 1
            if(predicted == torch.argmax(outputs)):
                running_accuracy += 1
 
        print('Accuracy of the model based on the test set of', test_split ,'inputs is:', (100*running_accuracy/total), '%'  )    

# Instance of the neural network, loss, optimizer 
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss() 

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


    
    X_im = data[:127500, :]
    X_im = np.load("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/X_im.csv.npy", "r")
    X_var = data[127500:127533, :]

    #X_im = resizeColouredImageCNN(X_im, 250, 170)
    #print("X_im Shape: ", np.shape(X_im))
    #np.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/X_im.csv", X_im, allow_pickle=False)
    #np.save("C:/Users/andre/OneDrive/Home PC desktop/Python/CRBot/X_im", X_im, allow_pickle=False)
    
    


    
    #print(X)

    print(X_im.shape)
    print(X_var.shape)
    
    Y = data[127533:127541, :]
    noVal = np.zeros(Y.shape[1])
    noVal = noVal.reshape(-1, 1)
    
    
    #print("Preprocess", Y.shape)
    counter = 0
    for column in Y.T:
        if np.sum(column) == 1:
            for num in range(len(column)):
                if column[num] == 1:
                    if np.sum(Y.T[counter - 1][:]) == 0:
                        Y.T[counter - 1][num] = 1
                    if counter + 1 >= Y.shape[1]:
                        #print("end")
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

    X_im = np.transpose(X_im)
    X_var = np.transpose(X_var)
    Y = np.transpose(Y)

    tensor_x_im = torch.Tensor(X_im) # transform to torch tensor
    tensor_x_var = torch.Tensor(X_var) 
    tensor_y = torch.Tensor(Y)

    tensor_x_im = tensor_x_im.transpose(1,3)

    print(tensor_x_im.shape)
    print(tensor_y.shape)

    my_dataset = TensorDataset(tensor_x_im, tensor_x_var, tensor_y) # create your datset


    train_size = int(0.8 * len(my_dataset))
    test_size = len(my_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset) # create your dataloader
    test_dataloader = DataLoader(test_dataset) # create your dataloader
    for epoch in range(10): # train for 10 epochs
        for batch in train_dataloader: 
            #print("batch = ", batch)
            x_im, x_var, y = batch
            x_im, x_var, y = x_im.to("cuda"), x_var.to("cuda"), y.to("cuda")
            yhat = clf(x_im, x_var)
            y =  torch.squeeze(y)
            if counter % 500 == 0:
                print(yhat)
                print(y)
            counter +=1
            #print("yhat=",yhat)
            #print("y=",y)
            loss = loss_fn(yhat, y)
            print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"--------------------Epoch:{epoch} loss is {loss.item()}-----------------------")
        with open('CNN_model_state.pt', 'wb') as f: 
            save(clf.state_dict(), f) 
        test(test_dataloader, int(0.8 * len(my_dataset)))

def predict(im, var):
    prm = ImageClassifier().to('cuda')
    with open('CNN_model_state.pt', 'rb') as f:
        prm.load_state_dict(load(f))
    prm.eval()
    tensorData_im = torch.tensor(im, dtype=torch.float32).to('cuda')
    tensorData_var = torch.tensor(var, dtype=torch.float32).to('cuda')
    return torch.nn.functional.softmax(prm(tensorData_im, tensorData_var), dim=0)




def predict_with_reinforcement_learning(im, var, selectedCard, reward):
    global var_saved, im_saved, yhatSaved, ySaved, rewardSaved
    prm = ImageClassifier().to('cpu')

    with open('CNN_model_state.pt', 'rb') as f:
        prm.load_state_dict(load(f))    

    tensorData_im = torch.tensor(im, dtype=torch.float32).to('cpu')
    tensorData_var = torch.tensor(var, dtype=torch.float32).to('cpu')

    yhat = torch.nn.functional.softmax(prm(tensorData_im, tensorData_var), dim=0)
    
    #if random.randint(0, 10) == 10:
        #yhat = 


    if selectedCard.sum() == 0:
        selectedCard = np.append(selectedCard, 1)
    else:
        selectedCard = np.append(selectedCard, 0)

    selectedCard = np.transpose(selectedCard)

    #if(len(var_saved) > 10):
        #var_saved.pop[0]
    #if(len(im_saved) > 10):
        #im_saved.pop[0]

    return yhat, tensorData_var, tensorData_im, yhat, torch.Tensor(selectedCard).to('cpu')
   

def apply_reinforcement_learning(im_saved, var_saved, yhatSaved, ySaved, rewardSaved):
    print("len", len(ySaved))
    prm = ImageClassifier().to('cpu')
    with open('CNN_model_state.pt', 'rb') as f:
        prm.load_state_dict(load(f))    
    opt = Adam(prm.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss() 
    for num in range(len(ySaved) - 3):
        #for a in range(min(max(10, num), 10)):
            #if num+1 < len(ySaved):
        reward = sum(rewardSaved[num: min(num+10, len(rewardSaved))])
        print("im", type(im_saved[num]))
        print("var", type(var_saved[num]))
        b = prm(im_saved[num], var_saved[num])
        print("learn:", b)
        print("rewardCard", reward)
        print("y", ySaved[num] )
        loss = loss_fn(b, ySaved[num][0]) * reward/500
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(num)
        with open('CNN_model_state.pt', 'wb') as f: 
            save(clf.state_dict(), f) 
    