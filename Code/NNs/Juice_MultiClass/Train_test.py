from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d,MaxUnpool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from tqdm import tqdm 
import model

model = model()
n_epochs = 801
loss_fn = nn.CrossEntropyLoss() 
loss_fn2 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.00009,  weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)


for epoch in range(n_epochs):
  correct_j = 0
  correct_d = 0
  sum = 0
  for dta, label_j, label_d in loader:
    torch.autograd.set_detect_anomaly(True)
    pred_j,pred_d = model(dta)
    loss11 = loss_fn2(pred_j, label_j.float())
    loss22 = loss_fn2(pred_d, label_d.float())
    loss =  loss11 + loss22
    correct_j += (torch.argmax(pred_j,1) == torch.argmax(label_j,1)).float().sum()
    correct_d += (torch.argmax(pred_d,1) == torch.argmax(label_d,1)).float().sum()
    sum += dta.shape[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  accuracy_j = 100 * correct_j / sum
  accuracy_d = 100 * correct_d / sum
  if epoch%10 == 0:
    print('Epoch: {} \tTraining Loss: {:.6f} \tAcc_J: {:.6f} \tAcc_D: {:.6f}'.format(epoch, loss, accuracy_j, accuracy_d))


batch_test = 1
loader_test = DataLoader(list(zip(test_dt, test_label_juice, test_label_day)), 
                         shuffle=False, batch_size=batch_test)

correct_j = 0
correct_d = 0
combined_acc = 0
sum = 0
pred_test_j = []
pred_test_d = []
label_test_j = []
label_test_d = []
for dta, label_j, label_d in tqdm(loader_test):
    pred_j, pred_d = model(dta)
    correct_j += (torch.argmax(pred_j,1) == torch.argmax(label_j,1)).float().sum()
    correct_d += (torch.argmax(pred_d,1) == torch.argmax(label_d,1)).float().sum()
    combined_acc += (torch.argmax(pred_j,1) == torch.argmax(label_j,1) and torch.argmax(pred_d,1) == torch.argmax(label_d,1) ).float().sum()
    sum += dta.shape[0]
    pred_test_j.append(np.array(pred_j.detach()).squeeze())
    pred_test_d.append(np.array(pred_d.detach()).squeeze())
    label_test_j.append(np.array(label_j).squeeze())
    label_test_d.append(np.array(label_d).squeeze())
accuracy_j = 100 * correct_j / sum
accuracy_d = 100 * correct_d / sum
combined_acc = 100* combined_acc/sum
print('Epoch: {} \tTesting Loss: {:.6f} \tAcc_j: {:.6f} \tAcc_d: {:.6f} \tAcc_combined: {:.6f}'.format(epoch, loss, accuracy_j, accuracy_d, combined_acc))


cm_j = confusion_matrix(np.argmax(np.array(pred_test_j),1),
                        np.argmax(np.array(label_test_j),1))
cm_d = confusion_matrix(np.argmax(np.array(pred_test_d),1),
                        np.argmax(np.array(label_test_d),1))


print("\t Confusion Matrix Juice \n", cm_j)
print("\t Confusion Matrix Day \n", cm_d)

class_0_p = []
class_0_t = []
class_1_p = []
class_1_t = []
class_2_p = []
class_2_t = []
class_3_p = []
class_3_t = []

for i in range(len(pred_test_j)):
  if np.argmax(pred_test_j,1)[i] == 0:
    class_0_p.append(pred_test_d[i])
    class_0_t.append( label_test_d[i])
  elif np.argmax(pred_test_j,1)[i] == 1:
    class_1_p.append(pred_test_d[i])
    class_1_t.append(label_test_d[i])
  elif np.argmax(pred_test_j,1)[i] == 2:
    class_2_p.append(pred_test_d[i])
    class_2_t.append( label_test_d[i])
  elif np.argmax(pred_test_j,1)[i] == 3:
    class_3_p.append(pred_test_d[i])
    class_3_t.append(label_test_d[i])
    
cm_0 = confusion_matrix(np.argmax(np.array(class_0_p),1), np.argmax(np.array(class_0_t),1))
print("\t Confusion Matrix for grape \n", cm_0)

cm_1 = confusion_matrix(np.argmax(np.array(class_1_p),1), np.argmax(np.array(class_1_t),1))
print("\t Confusion Matrix Juice for pineapple \n", cm_1)

cm_2 = confusion_matrix(np.argmax(np.array(class_2_p),1), np.argmax(np.array(class_2_t),1))
print("\t Confusion Matrix Juice for orange \n", cm_2)

cm_3 = confusion_matrix(np.argmax(np.array(class_3_p),1), np.argmax(np.array(class_3_t),1))
print("\t Confusion Matrix Juice for watermelon \n", cm_3)
