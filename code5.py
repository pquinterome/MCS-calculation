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

for i in range(5):
    a1 =feature_maps1[i, :, :]
    res1 = np.sum(a1, axis=2)
    res1 = [res1[w]/res1.max() for w in range(len(res1))]
    plt.figure(figsize=(8,4))
    plt.imshow(X_test1[i], cmap='Greys', alpha=0.7)
    plt.imshow(res1, cmap='jet', interpolation='nearest', alpha=0.3, vmin=0.6)
    cbar = plt.colorbar()
    cbar.set_label('Normalized activation map intensity', rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    #plt.colorbar()
    plt.xlabel('Control Points')
    plt.ylabel('Leaf Number')
    plt.title(f'Plan_{i} verification')
    plt.xlim(0,176)
    plt.savefig(f'output/M1_{i}.png', bbox_inches='tight')


    a2 =feature_maps2[i, :]
    res2 = np.sum(a2, axis=1)
    res2 = res2/res2.max()
    plt.figure(figsize=(8,4))
    x = np.arange(0.0, len(res2), 1)
    plt.plot(X_test2[i], alpha=1, linewidth=1.5, label='MUcp_profile')
    #plt.plot(res2)
    plt.fill_between(x= x, y1= X_test2[i].ravel(), y2= res2, color='gray', label='Activation zone', alpha=0.5, where= res2>0.7)
    plt.legend()
    plt.xlim(0,176)
    plt.savefig(f'output/Mcp_{i}.png', bbox_inches='tight')

    #a3 =feature_maps3[i, :, :]
    #res3 = np.sum(a3, axis=2)
    #plt.figure(figsize=(28,4))
    #plt.imshow(X_test3[i], cmap='Greys', alpha=0.7)
    #plt.contour(res3, cmap='jet', alpha=1)
    #plt.colorbar()
    #plt.savefig(f'output/CDI_{i}.png', bbox_inches='tight')

print('End of first Experiment')

val_ltm = np.load('inputs/tlm_val.npy')
val_ltm = np.array([val_ltm[i][:177,].T for i in range(len(val_ltm))])

y3 = pd.read_csv('inputs/id_val.csv')
y3['2_2'] = y3['2_2'].fillna(y3['2_2'].mean())
y = y3['2_2']/100
y_test = np.array([0 if x >= 0.98 else 1 for x in y])

ltm = val_ltm
a=[]
for i in range(len(ltm)):
    dlb1= pd.DataFrame(ltm[i].T)
    nunique1 = dlb1.apply(pd.Series.nunique)
    cols_to_drop1 = nunique1[nunique1 == 1].index
    a1= dlb1.drop(cols_to_drop1, axis=1)
    a.append(a1.T)
w = np.array([a[i].shape[0] for i in range(len(a))])
padded_array = np.zeros((w.max(), 177))
#padded_array = np.zeros((70, 177))
z=[]
for i in range(len(ltm)):
    q = a[i]
    shape = np.shape(q)
    padded_array = np.zeros((70, 177))
    padded_array[:shape[0],:shape[1]] = q
    padded_array.shape
    z.append(padded_array)
ltm = np.array(z)
mu = np.load('inputs/mu_val.npy')
mu = [mu[w]/mu[w].max() for w in range(len(mu))]
p = np.load('inputs/portal_val.npy', allow_pickle=True)
z=[]
for i in range(len(p)):
    q = p[i]
    shape = np.shape(q)
    padded_array = np.zeros((512, 512))
    padded_array[:shape[0],:shape[1]] = q
    padded_array.shape
    z.append(padded_array)
p = np.array(z)
print('LTM_dataset', ltm.shape)
print('MU_cp_dataset', mu.shape)
print('Portal dataset', p.shape)
print('labels', y_test.shape)

X_test1 = ltm.reshape(32, 70, 177, 1)
X_test2 = mu.reshape(32, 176, 1)
X_test3 = p.reshape(32, 512, 512, 1)


print('LTM_dataset', X_train1.shape)
print('MU_cp_dataset', X_train2.shape)
print('Portal dataset', X_train3.shape)
print('labels', y_test.shape)


model1 = tf.keras.models.load_model('models/model_1.h5')
model2 = tf.keras.models.load_model('models/model_2.h5')
model3 = tf.keras.models.load_model('models/model_3.h5')
model4 = tf.keras.models.load_model('models/model_4.h5')
model5 = tf.keras.models.load_model('models/model_5.h5')
model6 = tf.keras.models.load_model('models/model_6.h5')
model7 = tf.keras.models.load_model('models/model_7.h5')


print('End of first Experiment')

models= [model1, model1, model1, model1, model1]
print('all ok')
tprs1 = []
aucs1 = []
fprs1 = []
tprs2 = []
aucs2 = []
fprs2 = []
tprs3 = []
aucs3 = []
fprs3 = []
tprs4 = []
aucs4 = []
fprs4 = []
mean_fpr1 = np.linspace(0, 1, 100)
mean_fpr2 = np.linspace(0, 1, 100)
mean_fpr3 = np.linspace(0, 1, 100)
mean_fpr4 = np.linspace(0, 1, 100)

i = 1
fig1, ax1 = plt.subplots()
for model in models:
    #model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
    ##model.fit(x=[X_train1, X_train2, X_train3], y= y_train, validation_data= ([X_test1, X_test2, X_test3], y_test), epochs=200 ,verbose=0, callbacks=[early_stop, reduce_lr])
    #model.fit(x=X_train1, y= y_train, validation_data= (X_test1, y_test), epochs=200 ,verbose=0, callbacks=[early_stop, reduce_lr])
    ##model.fit_generator(train_generator, validation_data= test_generator, callbacks=[early_stop], epochs=200, verbose=0)
    model1.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
    model2.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
    model3.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
    model4.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])


    y_pred_keras = model1.predict(X_test1).ravel() 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_keras)
    tprs1.append(interp(mean_fpr1, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs1.append(roc_auc) 

    y_pred_keras2 = model2.predict(X_test2).ravel() 
    fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_keras2)
    tprs2.append(interp(mean_fpr2, fpr2, tpr2))
    roc_auc2 = auc(fpr2, tpr2)
    aucs2.append(roc_auc2)

    y_pred_keras3 = model3.predict(X_test3).ravel() 
    fpr3, tpr3, thresholds3 = roc_curve(y_test, y_pred_keras3)
    tprs3.append(interp(mean_fpr3, fpr3, tpr3))
    roc_auc3 = auc(fpr3, tpr3)
    aucs3.append(roc_auc3)

    y_pred_keras4 = model4.predict((X_test1, X_test2, X_test3)).ravel() 
    fpr4, tpr4, thresholds4 = roc_curve(y_test, y_pred_keras4)
    tprs4.append(interp(mean_fpr4, fpr4, tpr4))
    roc_auc4 = auc(fpr4, tpr4)
    aucs4.append(roc_auc4)
    

ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs1, axis=0)
mean_tpr2 = np.mean(tprs2, axis=0)
mean_tpr3 = np.mean(tprs3, axis=0)
mean_tpr4 = np.mean(tprs4, axis=0)
mean_tpr[-1] = 1.0
mean_tpr2[-1] = 1.0
mean_tpr3[-1] = 1.0
mean_tpr4[-1] = 1.0
mean_auc = np.mean(aucs1)
mean_auc2 = np.mean(aucs2)
mean_auc3 = np.mean(aucs3)
mean_auc4 = np.mean(aucs4)
std_auc = np.std(aucs1)
std_auc2 = np.std(aucs2)
std_auc3 = np.std(aucs3)
std_auc4 = np.std(aucs4)
ax1.plot(mean_fpr1, mean_tpr, color='blue', linestyle='-',label=r'M_1 ROC_AUC = %0.2f $\pm$ %0.2f' % (mean_auc, std_auc), lw=2, alpha=.7)
std_tpr = np.std(tprs1, axis=0)
ax1.plot(mean_fpr2, mean_tpr2, color='green' , linestyle='-.',label=r'M_2 ROC_AUC = %0.2f $\pm$ %0.2f' % (mean_auc2, std_auc2), lw=2, alpha=.7)
std_tpr2 = np.std(tprs2, axis=0)
ax1.plot(mean_fpr3, mean_tpr3, color='#F97306', linestyle=':',label=r'M_3 ROC_AUC = %0.2f $\pm$ %0.2f' % (mean_auc3, std_auc3), lw=2, alpha=.7)
std_tpr3 = np.std(tprs3, axis=0)
#ax1.plot(mean_fpr4, mean_tpr4, color='red',label=r'M_123 ROC_AUC = %0.2f $\pm$ %0.2f' % (mean_auc4, std_auc4), lw=2, alpha=.7)
std_tpr4 = np.std(tprs4, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
tprs_upper2 = np.minimum(mean_tpr2 + std_tpr2, 1)
tprs_lower2 = np.maximum(mean_tpr2 - std_tpr2, 0)
tprs_upper3 = np.minimum(mean_tpr3 + std_tpr3, 1)
tprs_lower3 = np.maximum(mean_tpr3 - std_tpr3, 0)
tprs_upper4 = np.minimum(mean_tpr4 + std_tpr4, 1)
tprs_lower4 = np.maximum(mean_tpr4 - std_tpr4, 0)


ax1.fill_between(mean_fpr1, tprs_lower, tprs_upper, color='blue', alpha=.2, label=r'$\pm$ 1 std. dev. M_1')
ax1.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='green', alpha=.2, label=r'$\pm$ 1 std. dev. M_2')
ax1.fill_between(mean_fpr3, tprs_lower3, tprs_upper3, color='orange', alpha=.2, label=r'$\pm$ 1 std. dev. M_3')
#ax1.fill_between(mean_fpr4, tprs_lower4, tprs_upper4, color='red', alpha=.2, label=r'$\pm$ 1 std. dev. M_123')

ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
ax1.legend(loc="right", bbox_to_anchor=(1.65, 0.5))

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('output/drop_01.png', bbox_inches='tight')
print('End of first Experiment')



models= [model1, model1, model1, model1, model1]
print('all ok')
tprs1 = []
aucs1 = []
fprs1 = []
tprs2 = []
aucs2 = []
fprs2 = []
tprs3 = []
aucs3 = []
fprs3 = []
tprs4 = []
aucs4 = []
fprs4 = []
mean_fpr1 = np.linspace(0, 1, 100)
mean_fpr2 = np.linspace(0, 1, 100)
mean_fpr3 = np.linspace(0, 1, 100)
mean_fpr4 = np.linspace(0, 1, 100)
#ltm = ltm[-411:]
#mu = mu[-411:]
#p = p[-411:]
#y = y[-411:]
#y2 = y2[-411:]
i = 1
fig1, ax1 = plt.subplots()
for model in models:
    #model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
    ##model.fit(x=[X_train1, X_train2, X_train3], y= y_train, validation_data= ([X_test1, X_test2, X_test3], y_test), epochs=200 ,verbose=0, callbacks=[early_stop, reduce_lr])
    #model.fit(x=X_train1, y= y_train, validation_data= (X_test1, y_test), epochs=200 ,verbose=0, callbacks=[early_stop, reduce_lr])
    ##model.fit_generator(train_generator, validation_data= test_generator, callbacks=[early_stop], epochs=200, verbose=0)
    model4.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
    model5.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
    model6.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
    model7.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])



    y_pred_keras4 = model4.predict((X_test1, X_test2, X_test3)).ravel() 
    fpr4, tpr4, thresholds4 = roc_curve(y_test, y_pred_keras4)
    tprs4.append(interp(mean_fpr4, fpr4, tpr4))
    roc_auc4 = auc(fpr4, tpr4)
    aucs4.append(roc_auc4)
    
    y_pred_keras = model5.predict((X_test1, X_test2)).ravel() 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_keras)
    tprs1.append(interp(mean_fpr1, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs1.append(roc_auc) 

    y_pred_keras2 = model6.predict((X_test1, X_test3)).ravel() 
    fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_keras2)
    tprs2.append(interp(mean_fpr2, fpr2, tpr2))
    roc_auc2 = auc(fpr2, tpr2)
    aucs2.append(roc_auc2)

    y_pred_keras3 = model7.predict((X_test2, X_test3)).ravel() 
    fpr3, tpr3, thresholds3 = roc_curve(y_test, y_pred_keras3)
    tprs3.append(interp(mean_fpr3, fpr3, tpr3))
    roc_auc3 = auc(fpr3, tpr3)
    aucs3.append(roc_auc3)
    

ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs1, axis=0)
mean_tpr2 = np.mean(tprs2, axis=0)
mean_tpr3 = np.mean(tprs3, axis=0)
mean_tpr4 = np.mean(tprs4, axis=0)
mean_tpr[-1] = 1.0
mean_tpr2[-1] = 1.0
mean_tpr3[-1] = 1.0
mean_tpr4[-1] = 1.0
mean_auc = np.mean(aucs1)
mean_auc2 = np.mean(aucs2)
mean_auc3 = np.mean(aucs3)
mean_auc4 = np.mean(aucs4)
std_auc = np.std(aucs1)
std_auc2 = np.std(aucs2)
std_auc3 = np.std(aucs3)
std_auc4 = np.std(aucs4)
ax1.plot(mean_fpr1, mean_tpr, color='blue',linestyle='-' ,label=r'M_123 ROC_AUC = %0.2f $\pm$ %0.2f' % (mean_auc, std_auc), lw=2, alpha=.7)
std_tpr = np.std(tprs1, axis=0)
ax1.plot(mean_fpr2, mean_tpr2, color='green', linestyle='--',label=r'M_12 ROC_AUC = %0.2f $\pm$ %0.2f' % (mean_auc2, std_auc2), lw=2, alpha=.7)
std_tpr2 = np.std(tprs2, axis=0)
ax1.plot(mean_fpr3, mean_tpr3, color='orange', linestyle='-.', label=r'M_13 ROC_AUC = %0.2f $\pm$ %0.2f' % (mean_auc3, std_auc3), lw=2, alpha=.7)
std_tpr3 = np.std(tprs3, axis=0)
ax1.plot(mean_fpr4, mean_tpr4, color='red',linestyle=':', label=r'M_23 ROC_AUC = %0.2f $\pm$ %0.2f' % (mean_auc4, std_auc4), lw=2, alpha=.7)
std_tpr4 = np.std(tprs4, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
tprs_upper2 = np.minimum(mean_tpr2 + std_tpr2, 1)
tprs_lower2 = np.maximum(mean_tpr2 - std_tpr2, 0)
tprs_upper3 = np.minimum(mean_tpr3 + std_tpr3, 1)
tprs_lower3 = np.maximum(mean_tpr3 - std_tpr3, 0)
tprs_upper4 = np.minimum(mean_tpr4 + std_tpr4, 1)
tprs_lower4 = np.maximum(mean_tpr4 - std_tpr4, 0)


ax1.fill_between(mean_fpr1, tprs_lower, tprs_upper, color='blue', alpha=.1, label=r'$\pm$ 1 std. dev. M_123')
ax1.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='green', alpha=.1, label=r'$\pm$ 1 std. dev. M_12')
ax1.fill_between(mean_fpr3, tprs_lower3, tprs_upper3, color='orange', alpha=.1, label=r'$\pm$ 1 std. dev. M_13')
ax1.fill_between(mean_fpr4, tprs_lower4, tprs_upper4, color='red', alpha=.1, label=r'$\pm$ 1 std. dev. M_23')

ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
ax1.legend(loc="right", bbox_to_anchor=(1.65, 0.5))

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('output/drop_02.png', bbox_inches='tight')


metrics5 = pd.DataFrame(model5.history.history)
pred1 = model5.predict((X_test1, X_test2))
predictions1 = np.round(pred1)
fpr1, tpr1, thresholds1 = roc_curve(y_test, pred1)
roc_auc1 = auc(fpr1, tpr1)
classes=[0,1]
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predictions1).numpy()
print(f'AUC_model{5}',  roc_auc1)
print(f'Accuracy{5}',   accuracy_score(y_test, predictions1))
print(f'precision{5}',  precision_score(y_test, predictions1))
print(f'recall{5}',     recall_score(y_test, predictions1))
print(f'f1{5}',         f1_score(y_test, predictions1))#

metrics6 = pd.DataFrame(model6.history.history)
pred2 = model6.predict((X_test1, X_test3))
predictions2 = np.round(pred2)
fpr2, tpr2, thresholds2 = roc_curve(y_test, pred2)
roc_auc2 = auc(fpr2, tpr2)
classes=[0,1]
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predictions2).numpy()
print(f'AUC_model{6}',  roc_auc2)
print(f'Accuracy{6}',   accuracy_score(y_test, predictions2))
print(f'precision{6}',  precision_score(y_test, predictions2))
print(f'recall{6}',     recall_score(y_test, predictions2))
print(f'f1{6}',         f1_score(y_test, predictions2))#

metrics7 = pd.DataFrame(model7.history.history)
pred3 = model7.predict((X_test2, X_test3))
predictions3 = np.round(pred3)
fpr3, tpr3, thresholds3 = roc_curve(y_test, pred3)
roc_auc3 = auc(fpr3, tpr3)
classes=[0,1]
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predictions3).numpy()
print(f'AUC_model{7}',  roc_auc3)
print(f'Accuracy{7}',   accuracy_score(y_test, predictions3))
print(f'precision{7}',  precision_score(y_test, predictions3))
print(f'recall{7}',     recall_score(y_test, predictions3))
print(f'f1{7}',         f1_score(y_test, predictions3))#


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

for i in range(5):
    a1 =feature_maps1[i, :, :]
    res1 = np.sum(a1, axis=2)
    res1 = [res1[w]/res1.max() for w in range(len(res1))]
    plt.figure(figsize=(8,4))
    plt.imshow(X_test1[i], cmap='Greys', alpha=0.7)
    plt.imshow(res1, cmap='jet', interpolation='nearest', alpha=0.3, vmin=0.6)
    cbar = plt.colorbar()
    cbar.set_label('Normalized activation map intensity', rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    #plt.colorbar()
    plt.xlabel('Control Points')
    plt.ylabel('Leaf Number')
    plt.title(f'Plan_{i} verification')
    plt.xlim(0,176)
    plt.savefig(f'output/M1_Validation{i}.png', bbox_inches='tight')


    a2 =feature_maps2[i, :]
    res2 = np.sum(a2, axis=1)
    res2 = res2/res2.max()
    plt.figure(figsize=(8,4))
    x = np.arange(0.0, len(res2), 1)
    plt.plot(X_test2[i], alpha=1, linewidth=1.5, label='MUcp_profile')
    #plt.plot(res2)
    plt.fill_between(x= x, y1= X_test2[i].ravel(), y2= res2, color='gray', label='Activation zone', alpha=0.5, where= res2>0.7)
    plt.legend()
    plt.xlim(0,176)
    plt.savefig(f'output/Mcp_Validation{i}.png', bbox_inches='tight')