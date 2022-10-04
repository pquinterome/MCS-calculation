#%%
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from numpy import asarray, interp, asarray
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.metrics import plot_roc_curve, auc, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, BatchNormalization, MaxPool1D ,MaxPool2D, Flatten, Dropout, GlobalMaxPooling2D, concatenate, SimpleRNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from scipy.stats import shapiro
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from numpy import interp
print('TensorFlow version', tf.__version__)

import random
# %%
ltm_T = np.load('inputs/ltm_T.npy')
ltm_H = np.load('inputs/ltm_H.npy')
ltm = np.concatenate((ltm_H, ltm_T), axis=0)
y = np.load('inputs/y.npy')
y = y[:820,]
y2 = np.load('inputs/y.npy')
y2 = y2[:820,]
y = np.array([0 if x >= 0.98 else 1 for x in y])
a=[]
for i in range(len(ltm)):
    dlb1= pd.DataFrame(ltm[i].T)
    nunique1 = dlb1.apply(pd.Series.nunique)
    cols_to_drop1 = nunique1[nunique1 == 1].index # hola
    a1= dlb1.drop(cols_to_drop1, axis=1)
    a.append(a1.T)
w = np.array([a[i].shape[0] for i in range(len(a))])
padded_array = np.zeros((w.max(), 177))
z=[]
for i in range(len(ltm)):
    q = a[i]
    shape = np.shape(q)
    padded_array = np.zeros((w.max(), 177))
    padded_array[:shape[0],:shape[1]] = q
    padded_array.shape
    z.append(padded_array)
ltm = np.array(z)
ltm = ltm [:820,:]
mu = np.load('inputs/mu_cp.npy')
mu = mu[:820,]
p = np.load('inputs/portal.npy')
ltm = np.concatenate((ltm, ltm[-411:]), axis=0)
p = np.concatenate((p, p[-411:]), axis=0)
mu= np.concatenate((mu, mu[-411:]), axis=0)
y = np.concatenate((y,y[-411:]), axis=0)
y2 = np.concatenate((y2,y2[-411:]), axis=0)

#print('dataset', ltm.shape)
#print('labels', y.shape)
#print('MU_cp', mu.shape)
#%%
X_train1, X_test1, X_train2, X_test2, X_train3, X_test3, y_train, y_test, y_train2, y_test2 = train_test_split(ltm, mu, p, y, y2, test_size=0.2)#, random_state= 35)
#print('X_train', X_train1.shape)
#print('X_test', X_test1.shape)
X_train1 = X_train1.reshape(984, 70, 177, 1)
X_test1  = X_test1.reshape(247, 70, 177, 1)
scaler= MinMaxScaler()
X_train2 = scaler.fit_transform(X_train2)
X_test2 = scaler.fit_transform(X_test2)
X_train2 = X_train2.reshape(984, 176, 1)
X_test2  = X_test2.reshape(247, 176, 1)
X_train3 = X_train3.reshape(984, 512, 512, 1)
X_test3 = X_test3.reshape(247, 512, 512, 1)
print('X_train1', X_train1.shape)
print('X_test1', X_test1.shape)
print('X_train2', X_train2.shape)
print('X_test2', X_test2.shape)
print('X_train3', X_train3.shape)
print('X_test3', X_test3.shape)
print('y_test', y_test.shape)
print('y_test2', y_test.shape)
print('X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X')
#%%
model1 = tf.keras.models.load_model('models/model_1.h5')
model2 = tf.keras.models.load_model('models/model_2.h5')
model3 = tf.keras.models.load_model('models/model_3.h5')
model4 = tf.keras.models.load_model('models/model_4.h5')
model5 = tf.keras.models.load_model('models/model_5.h5')
model6 = tf.keras.models.load_model('models/model_6.h5')
model7 = tf.keras.models.load_model('models/model_7.h5')
print('X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X')
#####################
y = ['pass' if y_test[i]>= 0.5 else 'fail' for i in range(30)]
a1 = model1.predict(X_test1)
a1 = ['pass' if a1[i]>= 0.5 else 'fail' for i in range(30)]
c1 = [ 'ok_'+y[i] if a1[i]== y[i] else 'fail' for i in range(30)]
a2 = model2.predict(X_test2)
a2 = ['pass' if a2[i]>= 0.5 else 'fail' for i in range(30)]
c2 = [ 'ok_'+y[i] if a2[i]== y[i] else 'fail' for i in range(30)]
a3 = model3.predict(X_test3)
a3 = ['pass' if a3[i]>= 0.5 else 'fail' for i in range(30)]
c3 = [ 'ok_'+y[i] if a3[i]== y[i] else 'fail' for i in range(30)]
##########################
print('X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X')
print('C1->', c1)
print('C2->', c2)
print('C3->', c3)
model1 = Model(inputs=model1.inputs, outputs=model1.layers[1].output)   #Sencod layer-> "layers[1]" is the first Conv layer
model2 = Model(inputs=model2.inputs, outputs=model2.layers[1].output)
model3 = Model(inputs=model3.inputs, outputs=model3.layers[1].output)


feature_maps1 = model1.predict(X_test1)
feature_maps2 = model2.predict(X_test2)
#feature_maps3 = model3.predict(X_test3)

print('X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X')

for i in range(30):
    a1 =feature_maps1[i, :, :]
    res1 = np.sum(a1, axis=2)
    res1 = [res1[w]/res1.max() for w in range(len(res1))]
    plt.figure(figsize=(8,4))
    plt.imshow(X_test1[i], cmap='Greys', alpha=0.7)
    plt.imshow(res1, cmap='jet', interpolation='nearest', alpha=0.3, vmin=0.5)
    cbar = plt.colorbar()
    cbar.set_label('Normalized activation map intensity', rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    #plt.colorbar()
    plt.xlabel('Control Points')
    plt.ylabel('Leaf Number')
    plt.title(f'Plan_{i} verification')
    plt.savefig(f'output/M1_{i}.png', bbox_inches='tight')




    a2 =feature_maps2[i, :]
    res2 = np.sum(a2, axis=1)
    res2 = res2/res2.max()
    plt.figure(figsize=(8,4))
    x = np.arange(0.0, len(res2), 1)
    plt.plot(X_test2[i], alpha=1, linewidth=5.5, label='MUcp_profile')
    #plt.plot(res2)
    plt.fill_between(x= x, y1= X_test2[i].ravel(), y2= res2, color='gray', label='Activation zone', alpha=0.5, where= res2>0.7)
    plt.legend()
    plt.savefig(f'output/Mcp_{i}.png', bbox_inches='tight')

    #a3 =feature_maps3[i, :, :]
    #res3 = np.sum(a3, axis=2)
    #plt.figure(figsize=(28,4))
    #plt.imshow(X_test3[i], cmap='Greys', alpha=0.7)
    #plt.contour(res3, cmap='jet', alpha=1)
    #plt.colorbar()
    #plt.savefig(f'output/CDI_{i}.png', bbox_inches='tight')




