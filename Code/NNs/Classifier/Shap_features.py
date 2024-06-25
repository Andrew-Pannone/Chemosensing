import shap
import model, train_dt, test_dt
import pickle
import pandas as  pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D,art3d
import pylab

'''Get Shap values with the trained model 
'''
shap_e = []
e = shap.DeepExplainer(model, train_dt)
shap_dt = e.shap_values(test_dt)

''' save - Test data, Shapley values, True label and Predicted labels
'''
path_shaply = " Location to save the files " 

with open(path_shaply + 'test_data', 'wb') as file:
     pickle.dump(test_dt, file)

with open(path_shaply + 'shap_values', 'wb') as file:
     pickle.dump(shap_dt, file)

pred_labels = np.argmax(pred_labels,1)
with open(path_shaply + 'pred_labels', 'wb') as file:
     pickle.dump(pred_labels, file)

true_labels = np.argmax(true_labels,1)
with open(path_shaply + 'true_labels', 'wb') as file:
     pickle.dump(true_labels, file)

'''Plot Shapely features with test data in 3D plots #Dummy Example with number of classes = 5
'''
num_classes = 5

def shap_batching(t):
    mask1 = t >= 0
    mask2 = t < 0
    mask3 = []
    for i in range(len(mask1)):
        mask3.append(~mask1[i] and ~mask2[i])

    count_plus = mask1[mask1 == True]
    count_plus = count_plus.shape[0]/202
    t1 = t[mask1]
    t2 = t[mask2]
    t1 = (t1 -  t1.min())/(t1.max() - t1.min())  
    t2 = (t2 -  t2.min())/(t2.max() - t2.min()) - 1 
    t[mask1] = t1
    t[mask2] = t2
    return t

# Indexing to match the test sample with shap values
o1 = 0
o2 = 4
o3 = 1
o4 = 2
o5 = 6

aa =  list(test_dt[o4]) + list(test_dt[o5]) + list(test_dt[o2]) + list(test_dt[o3]) + list(test_dt[o1])
bb =  list(np.array(list(range(0, 202)))) + list(np.array(list(range(0, 202)))) + list(np.array(list(range(0, 202)))) + list(np.array(list(range(0, 202)))) + list(np.array(list(range(0, 202))))
zz =  list(0*np.ones(202)) + list(4*np.ones(202)) + list(8*np.ones(202)) + list(12*np.ones(202)) + list(16*np.ones(202))
cc =  list(shap_batching(shap_dt[3][o4])) + list(shap_batching(shap_dt[4][o5])) + list(shap_batching(shap_dt[1][o2])) + list(shap_batching(shap_dt[2][o3])) + list(shap_batching(shap_dt[0][o1]))

''' Code for the 3D plot
'''
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig)
ax.scatter(bb, zz,aa,c = cc , cmap='RdBu', marker= '.',s=20 ,alpha=1.0)
ax.view_init(27, 247)
ax.set_xlabel('X', fontsize=20, labelpad = 10)
ax.set_ylabel('y', fontsize=20)
ax.set_yticklabels([])
ax.set_zlabel('z', fontsize=20)
ax.tick_params(axis='z', which='major', pad=10)
ax.grid(False)

def lims(mplotlims):
    scale = 1.021
    offset = (mplotlims[1] - mplotlims[0])*scale
    return mplotlims[1] - offset, mplotlims[0] + offset

xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
i = pylab.array([xlims[0], ylims[0], zlims[0]])
f = pylab.array([xlims[0], ylims[0], zlims[1]])
p = art3d.Line3DCollection(pylab.array([[i, f]]))
p.set_color('black')
ax.add_collection3d(p)

i2 = pylab.array([xlims[0], ylims[0], zlims[0]])
f2 = pylab.array([xlims[0], ylims[1], zlims[0]])
p2 = art3d.Line3DCollection(pylab.array([[i2, f2]]))
p2.set_color('black')
ax.add_collection3d(p2)

i3 = pylab.array([xlims[0], ylims[0], zlims[1]])
f3 = pylab.array([xlims[0], ylims[1], zlims[1]])
p3 = art3d.Line3DCollection(pylab.array([[i3, f3]]))
p3.set_color('black')
ax.add_collection3d(p3)

i4 = pylab.array([xlims[0], ylims[1], zlims[0]])
f4 = pylab.array([xlims[0], ylims[1], zlims[1]])
p4 = art3d.Line3DCollection(pylab.array([[i4, f4]]))
p4.set_color('black')
ax.add_collection3d(p4)

i5 = pylab.array([xlims[0], ylims[1], zlims[1]])
f5 = pylab.array([xlims[1], ylims[1], zlims[1]])
p5 = art3d.Line3DCollection(pylab.array([[i5, f5]]))
p5.set_color('black')
ax.add_collection3d(p5)


i6 = pylab.array([xlims[0], ylims[1], zlims[0]])
f6 = pylab.array([xlims[1], ylims[1], zlims[0]])
p6 = art3d.Line3DCollection(pylab.array([[i6, f6]]))
p6.set_color('black')
ax.add_collection3d(p6)

i7 = pylab.array([xlims[1], ylims[0], zlims[0]])
f7 = pylab.array([xlims[1], ylims[0], zlims[1]])
p7 = art3d.Line3DCollection(pylab.array([[i7, f7]]))
p7.set_color('black')
ax.add_collection3d(p7)

ax.grid(False)
hxfont = {'fontname':'Arial',
          'size' : 12}
hyfont = {'fontname':'Arial',
          'size' : 12}
hzfont = {'fontname':'Arial',
          'size' : 12}

ax.set_yticks([0,4, 8 ,12, 16])
ax.set_xticks([0,100,200])
ax.set_zticks([])

labels = [-0.2,0.6, 1.4]
ax.set_xticklabels(labels, **hxfont)

labels = [-0.2,0.6, 1.4]
ax.set_xticklabels(labels, **hxfont)

ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')

ax.set_facecolor([28/255, 190/255, 183/255])

plt.rcParams['figure.dpi']= 300
plt.savefig("Path to save the 3D plot")
