from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d,MaxUnpool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from sklearn.metrics import confusion_matrix

''' Model Initialization
'''
class conv1d_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv1d(1, 4, 2, stride=1)
        self.relu = nn.ReLU()
        self.acti = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.linear_pred = nn.Sequential(
            nn.Linear(4*201, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Sigmoid(),
        )
    def forward(self,x):
      x = x.view(x.shape[0], 1, x.shape[1])
      x1 = self.cnn(x)
      x1 = x1.view(x.shape[0], 4*201)
      # x2 = self.flatten(x1)
      x3 = self.linear_pred(x1)
      return x3
  
batch = 64
loader = DataLoader(list(zip(train_dt, train_label)), shuffle=True, batch_size=batch)

model = conv1d_model()
n_epochs = 800
loss_fn = nn.CrossEntropyLoss() 
loss_fn2 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001,  weight_decay=1e-5)

''' Model Training
'''
for epoch in range(n_epochs):
  correct = 0
  sum = 0
  for dta, label in loader:
    #print(dta.shape)
    pred = model(dta)
    #print(out)
    loss1 = loss_fn(pred, torch.argmax(label, 1))
    loss2 = loss_fn2(pred, label.float())
    loss =  loss2 #+ loss1
    correct += (torch.argmax(pred,1) == torch.argmax(label,1)).float().sum()
    sum += dta.shape[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  accuracy = 100 * correct / sum
  print('Epoch: {} \tTraining Loss: {:.6f} \tAccuracy: {:.6f}'.format(epoch, loss, accuracy))

''' Model Testing
'''

batch_test = 1
loader_test = DataLoader(list(zip(test_dt, test_label)), shuffle= False, batch_size=batch_test)

correct = 0
sum = 0
pred_test = []
label_test = []
test_save = []
for dta, label in loader_test:
    pred = model(dta.reshape(1,-1))
    correct += (torch.argmax(pred,1) == torch.argmax(label,1)).float().sum()
    sum += dta.shape[0]
    pred_test.append(np.array(pred.detach()).squeeze())
    label_test.append(np.array(label).squeeze())
    test_save.append(np.array(dta.detach().squeeze()))
accuracy = 100 * correct / sum
print('Epoch: {} \tTraining Loss: {:.6f} \tAccuracy: {:.6f}'.format(epoch, loss, accuracy))

''' Evaluation Metric
'''
test_save = np.array(test_save)
pred_test = np.array(pred_test)
label_test = np.array(label_test)
cm = confusion_matrix(np.argmax(pred_test,1),np.argmax(label_test,1))
