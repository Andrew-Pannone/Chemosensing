from skimage import util
from sklearn.metrics import mean_squared_error
import glob
import numpy as np
import random
import fnmatch
import os
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

from sklearn.utils import shuffle

''' Read files in Path1 and store class wise data 
'''

path1 = "Path to Data"
ls1 = os.listdir(path1)



def _get_device_data(file):
    '''
    Parameters
    ----------
    file : List
        List of all file names

    Returns
    -------
    p1 - p6 : List
        Return file names for each class
    '''
    
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    for i in range(len(file)):
        if file[i].rsplit('_')[3] == 'Class 1':
            p1.append(file[i])
        elif file[i].rsplit('_')[3] == 'Class 2':
            p2.append(file[i])
        elif file[i].rsplit('_')[3] == 'Class 3':
            p3.append(file[i])
        elif file[i].rsplit('_')[3] == 'Class 4':
            p4.append(file[i])
        elif file[i].rsplit('_')[3] == 'Class 5':
            p5.append(file[i])
        else:
            p6.append(file[i])
    return p1, p2, p3, p4, p5, p6

p1, p2, p3, p4, p5, p6 = _get_device_data(ls1)

''' Random Train test split per class
'''

train_p = random.sample(range(0, 175), int(0.7*175))
test_p = [i for i in range(0,175) if i not in train_p]


train_data = []
for i in range(len(train_p)):
  train_data.append(p1[train_p[i]])
  train_data.append(p2[train_p[i]])
  train_data.append(p3[train_p[i]])
  train_data.append(p4[train_p[i]])
  train_data.append(p5[train_p[i]])
  train_data.append(p6[train_p[i]])

train_data = shuffle(train_data, random_state= 0)

test_data = []
for i in range(len(test_p)):
  test_data.append(p1[test_p[i]])
  test_data.append(p2[test_p[i]])
  test_data.append(p3[test_p[i]])
  test_data.append(p4[test_p[i]])
  test_data.append(p5[test_p[i]])
  test_data.append(p6[test_p[i]])
  
test_data = shuffle(test_data, random_state= 0)

labl_dict = {'Class 1':0, 'Class 2':1, 'Class 3':2, 'Class 4':3, 'Class 5':4, 'Class 6':5 }

''' Read data into Train and Test
'''

def _get_data_label(file, l_dict):
  '''
    Parameters
    ----------
    file : List
        List of all file names
    l_dict : List
        List of corresponding Labels

    Returns
    -------
    t_dt : List
        Read files of Data
    tl : List
        Corresponding lables
    '''
  t_dt = []
  tl = []
  for i in range(len(file)):
    p1 = path1 + "/" + file[i]
    d1 = pd.read_csv(p1,skiprows = 116, usecols = [ 'Id' ])
    d1 = np.array(d1)
    d1 = (d1 - d1.mean())/(d1.std())
    t_dt.append(d1.reshape(-1,d1.shape[0]))
    
    tl.append(int(l_dict[file[i].rsplit('_')[3]]))
  return t_dt, tl

train_dt, train_lbl = _get_data_label(train_data, labl_dict)
train_dt = torch.FloatTensor(np.array(train_dt).squeeze())
train_label = np.eye(6)[train_lbl]
test_dt, test_lbl = _get_data_label(test_data, labl_dict)
test_dt = torch.FloatTensor(np.array(test_dt).squeeze())
test_label = np.eye(6)[test_lbl]
