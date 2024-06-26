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

'''Custom LambdaLayer to implement Cosine similarity metric
'''
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        
    def forward(self, x1,x2):
        return self.lambd(x1, x2)

'''Model Initialization
'''
class _model(nn.Module):
    def __init__(self):
         super().__init__()
         self.flatten = nn.Flatten()
         self.Sig = nn.Sigmoid()
         self.linear_relu_stack_1 = nn.Sequential(
             nn.Linear(202, 512),
             nn.ReLU(),
             nn.Linear(512, 1024),
             nn.ReLU(),
             )

         self.linear_relu_stack_2 = nn.Sequential(
             nn.Linear(202, 512),
             nn.ReLU(),
             nn.Linear(512, 1024),
             nn.ReLU(),
             )
         
         # self.cosine_layer = LambdaLayer(lambda x1, x2: np.dot(x1,x2)/(norm(x1)*norm(x2)))
         self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
         
         self.linear_relu_stack_3  = nn.Sequential(
             nn.Linear(1, 32),
             nn.ReLU(),
             nn.Linear(32, 1),
             nn.Tanh(),
             )
     
    def forward(self, x1, x2):
        batch = x1.shape[0]
        x1 = x1.view(batch, -1)
        x2 = x2.view(batch, -1)
        x1 = self.linear_relu_stack_1(x1)
        x2 = self.linear_relu_stack_2(x2)
        x3 = self.cos(x1,x2)
        x3 = x3.view(-1,1)
        x_out = self.linear_relu_stack_3(x3)
        return x_out, x3
