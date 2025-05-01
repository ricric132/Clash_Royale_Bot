import torch 
from torch import nn, save, load
from torch.optim import Adam
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
#from matplotlib import pyplot
from PIL import Image



class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1933, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 9)
        )

    def forward(self, x):
        return nn.functional.softmax(self.model(x))

if __name__ == "__main__":
    Tmodel = DNN().to("cuda")
    opt = Adam(Tmodel.parameters(), lr=0.0003)
    Tdata = np.random.rand(1, 1933)
    print(Tdata)
    Tmodel.load_state_dict(torch.load("model_state.pt"))
    print(torch.load("model_state.pt"))
    Tmodel.eval()
    dataTensor = torch.Tensor(Tdata).to("cuda")
    print(Tmodel(dataTensor))
    #print(Tmodel.state_dict())
