from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d,MaxUnpool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from skimage import util
from torchvision import transforms, models
from torch import optim

import train_dt, train_label_juice, train_label_day,batch

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv1d(1, 4, 2, stride=1)
        self.relu = nn.ReLU()
        self.acti = nn.Sigmoid()
        self.drop = nn.AlphaDropout(p=0.2)
        self.flatten = nn.Flatten()
        
        self.linear_pred_juice = nn.Sequential(
            nn.Linear(4*201, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

        self.linear_pred_day = nn.Sequential(
            nn.Linear(4*201, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.cnn(x) #(x, (h,c)), , (h,c)
        x = x.view(x.shape[0], 4*201)
        x1 = self.relu(x)
        x_juice = self.linear_pred_juice(x1)
        x_day = self.linear_pred_day(x1)
        return x_juice, x_day


batch = 32
loader = DataLoader(list(zip(train_dt, train_label_juice, train_label_day)), 
                    shuffle=True, batch_size=batch)
