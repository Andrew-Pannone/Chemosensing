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

import dataloader_train, dataloader_test

path_model = "Path to save model "
iters = " Number of interations "

''' _run_model definition for Training and testing 
'''
class _run_model():
    def __init__(self,path_model, iters ):
         super().__init__()
         self.path_model = path_model
         self.iters = iters
         self.model = _model()
         self.save_path = self.path_model + '\\' + 'model_gradient'
         self.loss_fn = nn.MSELoss()
         self.optimizer = torch.optim.Adam(self.model.parameters(), lr= 0.0001,  weight_decay=1e-5)
         
    def save_model(self):
         torch.save(self.model, self.save_path)
         
    def load_model(self):
        return torch.load(self.save_path)
        
    def export_files(self, file1, name):
        np.savetxt(self.path_model  + '\\' + name, np.array(file1).squeeze(), delimiter =',',fmt= '%s')

    def train(self, loader1):
        for epoch in range(self.iters):
            # _correct = 0
            # _sum = 0
            for dta1, dta2, label, ph_init in tqdm(loader1):
                out1, _ = self.model(dta1.float(), dta2.float())
                loss = self.loss_fn(6*out1.float().squeeze(), label.float().squeeze())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch%2 == 0:
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, loss))
        self.save_model()
    
    def test(self, loader2, code):
        model_test = self.load_model()
        pred_val = []
        true_val = []
        right_ph = []
        ls = 0
        _sum = 0
        
        for dta1, dta2, label, ph_init in tqdm(loader2):
            
            out1, _ = model_test(dta1.float(), dta2.float())
            
            loss = self.loss_fn(6*out1.float().squeeze(), label.float().squeeze())
            ls += loss
            _sum += dta1.shape[0]
            pred_val.append(np.array(6*out1.detach()))
            true_val.append(np.array(label))
            right_ph.append(np.array(ph_init))
        ls = ls/_sum
        print('\tTesting Loss: {:.6f}'.format(ls))
        self.export_files(pred_val, "pred_" +str(code) + ".csv")
        self.export_files(true_val, "true_" +str(code) + ".csv")
        self.export_files(right_ph, "Ph_gd_" +str(code) + ".csv")
    
_run = _run_model(path_model, iters)
print("Begin Training....")
_run.train(dataloader_train)
print("Begin Testing....")
_run.test(dataloader_test, 'custom marker')
