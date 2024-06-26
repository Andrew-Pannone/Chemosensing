import glob
import numpy as np
import random
import fnmatch
import os
from PIL import Image
from matplotlib import pyplot as plt
import PIL
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.preprocessing import StandardScaler as scl
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import norm
from sklearn.utils import shuffle
import os
import argparse
from tqdm import tqdm

path_buffer = " Path to pH Data "
train_file_path = " Path to train filenames "
path_inter = " Path to Intermediate pH value data " 
test_file_path = "Path to test filenames "
path_model = "Path to save model "
iters = " Number of interations "

''' Read files and perform preprocessing using _data_set_helper
'''
class _data_set_helper():
    def __init__(self, path1,path2):
        super().__init__()
        self.path1 = path1
        self.path2 = path2

    def _read_file(self, file1, file2):
        f11 = ('_').join(file1.rsplit('_')[1:])
        f22 = ('_').join(file2.rsplit('_')[1:])
        
        p11 = file1.rsplit('_')[0]
        p22 = file2.rsplit('_')[0]
        
        if p11 == 'Buffer':
            f1 = self.path1 + '\\' + f11        
        else:
            f1 = self.path2 + '\\' + f11
        
        if p22 == 'Inter':
            f2 = self.path2 + '\\' + f22
        else:
            f2 = self.path1 + '\\' + f22
            
        d1 = pd.read_csv(f1, skiprows = 114, usecols = ['Id'])
        d2 = pd.read_csv(f2, skiprows = 114, usecols = ['Id'])
        d1 = np.array(d1)
        d2 = np.array(d2)
        return d1,d2
    
    def _normalize_mm(self, file1, file2):
        d1, d2 = self._read_file(file1, file2)       
        d1 = (d1 - d1.min())/(d1.max() - d1.min())
        d2 = (d2 - d2.min())/(d2.max() - d1.min())
        return d1.reshape(-1,d1.shape[0]), d2.reshape(-1,d2.shape[0])
    
    def _normalize_z(self, file1, file2):
        d1, d2 = self._read_file(file1, file2)       
        d1 = (d1 - d1.mean())/(d1.std())
        d2 = (d2 - d2.mean())/(d2.std())
        return d1.reshape(-1,d1.shape[0]), d2.reshape(-1,d2.shape[0])

_dt_helper = _data_set_helper(path_buffer, path_inter)  

''' Store Train and test data perform transforms using Data generator - _data_gen
'''

class _data_gen(Dataset):
    def __init__(self, all_files, transform ):
        self.all_files = all_files
        self.transform = transform
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):

        item1 = self.all_files['0'][index]
        item2 = self.all_files['1'][index]
        item3 = self.all_files['2'][index]
        item4 = self.all_files['1'][index].rsplit('_')[3][2:]
        
        out_dt1, out_dt2 = _dt_helper._normalize_mm(item1, item2) 
        label = item3.astype(float)
        return self.transform(out_dt1), self.transform(out_dt2), label, item4

all_files_tr = pd.read_csv(train_file_path)
all_files_te = pd.read_csv(test_file_path)
train_data = _data_gen( all_files = all_files_tr, transform = transforms.Compose([ transforms.ToTensor()]))
test_data = _data_gen( all_files = all_files_te, transform = transforms.Compose([ transforms.ToTensor()]))

dataloader_train = torch.utils.data.DataLoader(train_data, batch_size = 128, shuffle= True)
dataloader_test = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle= True)
