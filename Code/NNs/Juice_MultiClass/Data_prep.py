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

from sklearn.utils import shuffle


path1 = "Path to Day_1"
ls1 = os.listdir(path1)

path2 = "Path to Day_2"
ls2 = os.listdir(path2)

path3 = "Path to Day_3"
ls3 = os.listdir(path3)

path4 = "Path to Day_4"
ls4 = os.listdir(path4)


split_ls1 = ls1[0].split('_')


d1 = []
d2 = []
d3 = []
d4 = []
def _get_device_data(file, d1, d2, d3, d4):
  for i in range(len(file)):
    if file[i].rsplit('_')[0] == 'Grape':
      d1.append([file[i], int(file[i].rsplit('_')[3][-1])])
    elif file[i].rsplit('_')[0] == 'Orange':
      d2.append([file[i], int(file[i].rsplit('_')[3][-1])])
    elif file[i].rsplit('_')[0] == 'Pineapple':
      d3.append([file[i], int(file[i].rsplit('_')[3][-1])])
    elif file[i].rsplit('_')[0] == 'Watermelon':
        if len(d4) < 1575:
            d4.append([file[i], int(file[i].rsplit('_')[3][-1])])

  return d1, d2, d3, d4

d1, d2, d3, d4 = _get_device_data(ls1, d1, d2, d3, d4)  # 1575
d1, d2, d3, d4 = _get_device_data(ls2, d1, d2, d3, d4)  # 1600   
d1, d2, d3, d4 = _get_device_data(ls3, d1, d2, d3, d4)  # 1600
d1, d2, d3, d4 = _get_device_data(ls4, d1, d2, d3, d4)  # 1577

print(len(d1),len(d2),len(d3),len(d4))

train_d1 = random.sample(range(0, len(d1)), int(0.7*len(d1)))
train_d2 = random.sample(range(0, len(d2)), int(0.7*len(d2)))
train_d3 = random.sample(range(0, len(d3)), int(0.7*len(d3)))
train_d4 = random.sample(range(0, len(d4)), int(0.7*len(d4)))

test_d1 = [i for i in range(0,len(d1)) if i not in train_d1]
test_d2 = [i for i in range(0,len(d2)) if i not in train_d2]
test_d3 = [i for i in range(0,len(d3)) if i not in train_d3]
test_d4 = [i for i in range(0,len(d4)) if i not in train_d4]

train_data = []

for i in range(len(train_d1)):
  train_data.append(d1[train_d1[i]])
for i in range(len(train_d2)):
  train_data.append(d2[train_d2[i]])
for i in range(len(train_d3)):
  train_data.append(d3[train_d3[i]])
for i in range(len(train_d4)):
  train_data.append(d4[train_d4[i]])

train_data = shuffle(train_data, random_state= 0)

test_data = []

for i in range(len(test_d1)):
  test_data.append(d1[test_d1[i]])
for i in range(len(test_d2)):
  test_data.append(d2[test_d2[i]])
for i in range(len(test_d3)):
  test_data.append(d3[test_d3[i]])
for i in range(len(test_d4)):
  test_data.append(d4[test_d4[i]])


def _get_data_label(file, path1, path2, path3, path4):
  t_dt = []
  tl1 = []
  tl2 = []
  for i in range(len(file)):
    if file[i][0].rsplit('_')[3][-1] == '1':
        p1 = path1 + "/" + file[i][0]
    elif file[i][0].rsplit('_')[3][-1] == '2':
        p1 = path2 + "/" + file[i][0]
    elif file[i][0].rsplit('_')[3][-1] == '3':
        p1 = path3 + "/" + file[i][0]
    elif file[i][0].rsplit('_')[3][-1] == '4':
        p1 = path4 + "/" + file[i][0]
    
    d1 = pd.read_csv(p1,skiprows = 114, usecols = [ 'Id' ])#delimiter= '\t',  encoding= 'unicode_escape')
    d1 = np.array(d1)
    if np.isnan(d1).sum() != 0:
        print(p1, np.isnan(d1).sum())
    #d1 = (d1 - d1.mean())/(d1.std())
    d1 = (d1 - d1.min())/(d1.max() - d1.min())
    t_dt.append(d1.reshape(-1,d1.shape[0]))

    if file[i][0].rsplit('_')[0] == 'Grape':
      tl1.append(0)
      tl2.append(file[i][1] - 1)
    elif file[i][0].rsplit('_')[0] == 'Pineapple':
      tl1.append(1)
      tl2.append(file[i][1] - 1)
    elif file[i][0].rsplit('_')[0] == 'Orange':
      tl1.append(2)
      tl2.append(file[i][1] - 1)
    elif file[i][0].rsplit('_')[0] == 'Watermelon':
      tl1.append(3)
      tl2.append(file[i][1] - 1)

  return t_dt, tl1, tl2


train_dt, train_lbl_juice, train_lbl_day = _get_data_label(train_data, path1, 
                                                           path2, path3, path4)
train_dt = torch.FloatTensor(np.array(train_dt).squeeze())
train_label_juice = np.eye(4)[train_lbl_juice]
train_label_day = np.eye(4)[train_lbl_day]


test_dt, test_lbl_juice,test_lbl_day  = _get_data_label(test_data, 
                                                        path1, path2, path3, path4)
test_dt = torch.FloatTensor(np.array(test_dt).squeeze())
test_label_juice = np.eye(4)[test_lbl_juice]
test_label_day = np.eye(4)[test_lbl_day]
