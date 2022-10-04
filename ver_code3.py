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
    cols_to_drop1 = nunique1[nunique1 == 1].index
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
p = np.array([(p[i]/p[i].max()) for i in range(len(p))])
#########################################################
#ltm = np.concatenate((ltm, ltm[-411:]), axis=0)
#p = np.concatenate((p, p[-411:]), axis=0)
#mu= np.concatenate((mu, mu[-411:]), axis=0)
#y = np.concatenate((y,y[-411:]), axis=0)
#y2 = np.concatenate((y2,y2[-411:]), axis=0)

#print('dataset', ltm.shape)
#print('labels', y.shape)
#print('MU_cp', mu.shape)
#%%
#X_train1, X_test1, X_train2, X_test2, X_train3, X_test3, y_train, y_test, y_train2, y_test2 = train_test_split(ltm, mu, p, y, y2, test_size=0.2, random_state= 35)
##print('X_train', X_train1.shape)
##print('X_test', X_test1.shape)
#X_train1 = X_train1.reshape(984, 70, 177, 1)
#X_test1  = X_test1.reshape(247, 70, 177, 1)
##X_train1 = X_train1.reshape(656, 70, 177, 1)
##X_test1  = X_test1.reshape(164, 70, 177, 1)
#scaler= MinMaxScaler()
#X_train2 = scaler.fit_transform(X_train2)
#X_test2 = scaler.fit_transform(X_test2)
#X_train2 = X_train2.reshape(984, 176, 1)
#X_test2  = X_test2.reshape(247, 176, 1)
#X_train3 = X_train3.reshape(984, 512, 512, 1)
#X_test3 = X_test3.reshape(247, 512, 512, 1)
##X_train2 = X_train2.reshape(656, 176, 1)
##X_test2  = X_test2.reshape(164, 176, 1)
##X_train3 = X_train3.reshape(656, 512, 512, 1)
##X_test3 = X_test3.reshape(164, 512, 512, 1)
#print('X_train1', X_train1.shape)
#print('X_test1', X_test1.shape)
#print('X_train2', X_train2.shape)
#print('X_test2', X_test2.shape)
#print('X_train3', X_train3.shape)
#print('X_test3', X_test3.shape)
#print('y_test', y_test.shape)
#print('y_test2', y_test.shape)
print('X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X')
#%%
activation = 'sigmoid' 
##softmax
##models---->>>
##

##1 Single layers
##Model->1
i1 = Input(shape=(70,177,1))
x1 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(i1)
x1 = MaxPool2D(pool_size=(2,2))(x1)
x1 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(x1)
x1 = MaxPool2D(pool_size=(2,2))(x1)                                                
x1 = Dropout(rate=0.2)(x1)
x1 = Flatten()(x1)
#x1 = BatchNormalization()(x1)
x11 = Dense(90, activation='relu')(x1)
x11 = Dense(1, activation='sigmoid')(x11)
model1 = Model(i1, x11)
##Model->2
i2 = Input(shape=(176,1))
x2 = Conv1D(filters=70, kernel_size=(5), activation='relu', padding='same')(i2)
x2 = MaxPool1D(pool_size=(3))(x2)
x2 = Dropout(rate=0.1)(x2)
x2 = Flatten()(x2)
x2 = BatchNormalization()(x2)
x22 = Dense(90, activation='relu')(x2)
x22 = Dense(1, activation='sigmoid')(x22)
model2 = Model(i2, x22)
###Model->3
i3 = Input(shape=(512,512,1))
x3 = Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same')(i3)
x3 = MaxPool2D(pool_size=(3,3))(x3)
x3 = Dropout(0.1)(x3)
x3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x3)
x3 = MaxPool2D(pool_size=(2,2))(x3)
x3 = Dropout(0.1)(x3)
x3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x3)
x3 = MaxPool2D(pool_size=(2,2))(x3)
x3 = Dropout(0.2)(x3)
x3 = Flatten()(x3)
#x3 = BatchNormalization()(x3)
x33 = Dense(360, activation='relu')(x3)
x33 = Dense(90, activation='relu')(x33)
x33 = Dense(1, activation='sigmoid')(x33)
model3 = Model(i3, x33)
#####################
#merge
merge = concatenate([x1, x2, x3])
#merge = concatenate([x2, x3])
x = Dense(360, activation='relu')(merge)
x = Dense(180, activation='relu')(x)
#x = Dense(90, activation='relu')(x)
g1 = Dense(1, activation='sigmoid')(x)  #Classification Hybrid model
#x2 = Dense(1, activation='linear')(x)
model4 = Model(inputs=[i1, i2, i3], outputs=g1)

merge = concatenate([x1, x2])
x = Dense(360, activation='relu')(merge)
x = Dense(180, activation='relu')(x)
g1 = Dense(1, activation='sigmoid')(x)
model5 = Model(inputs=[i1, i2], outputs=g1)

merge = concatenate([x1, x3])
x = Dense(360, activation='relu')(merge)
x = Dense(180, activation='relu')(x)
g1 = Dense(1, activation='sigmoid')(x)
model6 = Model(inputs=[i1, i3], outputs=g1)

merge = concatenate([x2, x3])
x = Dense(360, activation='relu')(merge)
x = Dense(180, activation='relu')(x)
g1 = Dense(1, activation='sigmoid')(x)
model7 = Model(inputs=[i2, i3], outputs=g1)





# %%
#y_cat_train = to_categorical(y_train, 2)
#y_cat_test = to_categorical(y_test, 2)

##data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=10)
##train_generator = data_generator.flow(X_train3, y_train,  batch_size=64)
##test_generator = data_generator.flow(X_test3, y_test, shuffle=False)

#data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
#train_generator = data_generator.flow(X_train, y_train)
#test_generator = data_generator.flow(X_test, y_test, shuffle=False)
#train_generator = data_generator.flow([X_train1, X_train2], y_train)
#test_generator = data_generator.flow([X_test1, X_test2], y_test, shuffle=False)

early_stop = EarlyStopping(monitor='val_loss', patience=15)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=10, min_lr=0.00001)


#model1.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
#model2.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
#model3.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
#model1.fit(x=X_train1, y= y_train, validation_data= (X_test1, y_test), epochs=400 ,verbose=0, callbacks=[early_stop, reduce_lr])
#model2.fit(x=X_train2, y= y_train, validation_data= (X_test2, y_test), epochs=400 ,verbose=0, callbacks=[reduce_lr])
#model3.fit(x=X_train3, y= y_train, validation_data= (X_test3, y_test), epochs=400 ,verbose=0, callbacks=[early_stop, reduce_lr])

#model1.save('models/model_1.h5')
#model2.save('models/model_2.h5')
#model3.save('models/model_3.h5')

#model4.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
#model4.fit(x=[X_train1, X_train2, X_train3], y= y_train, validation_data= ([X_test1, X_test2, X_test3], y_test), epochs=200 ,verbose=0, callbacks=[early_stop, reduce_lr])
#model4.save('models/model_4.h5')
#model5.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
#model5.fit(x=[X_train1, X_train2], y= y_train, validation_data= ([X_test1, X_test2], y_test), epochs=200 ,verbose=0, callbacks=[early_stop, reduce_lr])
#model5.save('models/model_5.h5')
#model6.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
#model6.fit(x=[X_train1, X_train3], y= y_train, validation_data= ([X_test1, X_test3], y_test), epochs=200 ,verbose=0, callbacks=[early_stop, reduce_lr])
#model6.save('models/model_6.h5')
#model7.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
#model7.fit(x=[X_train2, X_train3], y= y_train, validation_data= ([X_test2, X_test3], y_test), epochs=200 ,verbose=0, callbacks=[early_stop, reduce_lr])
#model7.save('models/model_7.h5')
#model5.save('models/model_5.h5')
#model6.save('models/model_6.h5')
#model7.save('models/model_7.h5')


model1 = tf.keras.models.load_model('models/model_1.h5')
model2 = tf.keras.models.load_model('models/model_2.h5')
model3 = tf.keras.models.load_model('models/model_3.h5')
model4 = tf.keras.models.load_model('models/model_4.h5')
model5 = tf.keras.models.load_model('models/model_5.h5')
model6 = tf.keras.models.load_model('models/model_6.h5')
model7 = tf.keras.models.load_model('models/model_7.h5')




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
#ltm = ltm[-205:]
#mu = mu[-205:]
#p = p[-205:]
#y = y[-205:]
#y2 = y2[-205:]
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

    idx = np.random.choice(np.arange(len(p)), 205, replace=False)
    ltm1 = ltm[idx]
    mu1 = mu[idx]
    p1 = p[idx]
    y1 = y[idx]
    y21 = y2[idx]

    X_train0, X_test0, X_train2, X_test2, X_train3, X_test3, y_train, y_test, y_train2, y_test2 = train_test_split(ltm1, mu1, p1, y1, y21, test_size=0.2)
    print('X_train', X_train0.shape)
    print('X_test', X_test0.shape)
    X_train1 = X_train0.reshape(X_train0.shape[0], 70, 177, 1)
    X_test1  = X_test0.reshape(X_test0.shape[0], 70, 177, 1)
    scaler= MinMaxScaler()
    X_train2 = scaler.fit_transform(X_train2)
    X_test2 = scaler.fit_transform(X_test2)
    X_train2 = X_train2.reshape(X_train0.shape[0], 176, 1)
    X_test2  = X_test2.reshape(X_test0.shape[0], 176, 1)
    X_train3 = X_train3.reshape(X_train0.shape[0], 512, 512, 1)
    X_test3 = X_test3.reshape(X_test0.shape[0], 512, 512, 1)

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
ax1.plot(mean_fpr1, mean_tpr, color='blue', linestyle='--',label=r'M_1 ROC_AUC = %0.2f $\pm$ %0.2f' % (mean_auc, std_auc), lw=2, alpha=.7)
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






#models = [model3, model3, model3, model3, model3]
#print('all ok')
#tprs1 = []
#aucs1 = []
#fprs1 = []
#mean_fpr = np.linspace(0, 1, 100)
#i = 1
#fig1, ax1 = plt.subplots()
#for model in models:
#    model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
#    #model.fit(x=[X_train1, X_train2, X_train3], y= y_train, validation_data= ([X_test1, X_test2, X_test3], y_test), epochs=200 ,verbose=0, callbacks=[early_stop, reduce_lr])
#    model.fit(x=X_train3, y= y_train, validation_data= (X_test3, y_test), epochs=400 ,verbose=0, callbacks=[early_stop, reduce_lr])
#    #model.fit_generator(train_generator, validation_data= test_generator, callbacks=[early_stop], epochs=200, verbose=0)
#  
#    y_pred_keras = model.predict(X_test3).ravel() 
#    fpr, tpr, thresholds = roc_curve(y_test, y_pred_keras)
#    tprs1.append(interp(mean_fpr, fpr, tpr))
#    roc_auc = auc(fpr, tpr)
#    aucs1.append(roc_auc)    
#
#ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
#mean_tpr = np.mean(tprs1, axis=0)
#mean_tpr[-1] = 1.0
#mean_auc = np.mean(aucs1)
#std_auc = np.std(aucs1)
#ax1.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
#std_tpr = np.std(tprs1, axis=0)
#tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
#ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic DBPD")
#ax1.legend(loc="right", bbox_to_anchor=(1.65, 0.5))
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.savefig('output/drop_00.png', bbox_inches='tight')


metrics1 = pd.DataFrame(model1.history.history)
pred1 = model1.predict(X_test1)
predictions1 = np.round(pred1)
fpr1, tpr1, thresholds1 = roc_curve(y_test, pred1)
roc_auc1 = auc(fpr1, tpr1)
classes=[0,1]
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predictions1).numpy()
print(f'AUC_model{1}',  roc_auc1)
print(f'Accuracy{1}',   accuracy_score(y_test, predictions1))
print(f'precision{1}',  precision_score(y_test, predictions1))
print(f'recall{1}',     recall_score(y_test, predictions1))
print(f'f1{1}',         f1_score(y_test, predictions1))#

metrics2 = pd.DataFrame(model2.history.history)
pred2 = model2.predict(X_test2)
predictions2 = np.round(pred2)
fpr2, tpr2, thresholds2 = roc_curve(y_test, pred2)
roc_auc2 = auc(fpr2, tpr2)
classes=[0,1]
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predictions2).numpy()
print(f'AUC_model{2}',  roc_auc2)
print(f'Accuracy{2}',   accuracy_score(y_test, predictions2))
print(f'precision{2}',  precision_score(y_test, predictions2))
print(f'recall{2}',     recall_score(y_test, predictions2))
print(f'f1{2}',         f1_score(y_test, predictions2))#

metrics3 = pd.DataFrame(model3.history.history)
pred3 = model3.predict(X_test3)
predictions3 = np.round(pred3)
fpr3, tpr3, thresholds3 = roc_curve(y_test, pred3)
roc_auc3 = auc(fpr3, tpr3)
classes=[0,1]
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predictions3).numpy()
print(f'AUC_model{3}',  roc_auc3)
print(f'Accuracy{3}',   accuracy_score(y_test, predictions3))
print(f'precision{3}',  precision_score(y_test, predictions3))
print(f'recall{3}',     recall_score(y_test, predictions3))
print(f'f1{3}',         f1_score(y_test, predictions3))#

metrics4 = pd.DataFrame(model4.history.history)
pred4 = model4.predict((X_test1, X_test2, X_test3))
predictions4 = np.round(pred4)
fpr4, tpr4, thresholds4 = roc_curve(y_test, pred4)
roc_auc4 = auc(fpr4, tpr4)
classes=[0,1]
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predictions4).numpy()
print(f'AUC_model{4}',  roc_auc4)
print(f'Accuracy{4}',   accuracy_score(y_test, predictions4))
print(f'precision{4}',  precision_score(y_test, predictions4))
print(f'recall{4}',     recall_score(y_test, predictions4))
print(f'f1{4}',         f1_score(y_test, predictions4))#

print('now hybrid')




#i1 = Input(shape=(70,177,1))
#x1 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(i1)
#x1 = MaxPool2D(pool_size=(2,2))(x1)
#x1 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(x1)
#x1 = MaxPool2D(pool_size=(2,2))(x1)                                                
#x1 = Dropout(rate=0.2)(x1)
#x1 = Flatten()(x1)
##x1 = BatchNormalization()(x1)
#x1 = Dense(90, activation='relu')(x1)
#x1 = Dense(1, activation='linear')(x1)
#model6 = Model(i1, x1)


#model6.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])


#models = [model6, model6, model6, model6]
#i = 0
#for model in models:#
#
#    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
#    #model.fit(x=[X_train1, X_train2, X_train3], y= y_train2, validation_data= ([X_test1, X_test2, X_test3], y_test2), callbacks=[reduce_lr], epochs=200, verbose=0)
#    model.fit(x=X_train1, y= y_train2, validation_data= (X_test1, y_test2), callbacks=[reduce_lr], epochs=400, verbose=0)
#    #pred2 = model4.predict((X_test1, X_test2, X_test3))
#    model6.save(f'models/model_6_{i}.h5')
#    pred2 = model.predict(X_test1)
#    mae = mean_absolute_error(y_test2, pred2)
#    rmse = mean_squared_error(y_test2, pred2)
#    print(f'MAE{i}', mae)
#    print(f'RMSE{i}', rmse)
#    print(f'y_test2{i}>>>','', np.array(y_test2))
#    print(f'pred2>>>{i}','', np.array(pred2.ravel()))#
#
#    plt.figure(4+i)
#    plt.scatter(x=y_test2, y=pred2, edgecolors='k', color='g', alpha=0.7, label='Predict')
#    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
#    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='0%', alpha=.8)
#    plt.plot([0.03, 1], [0, 0.97], 'g--', linewidth=0.8)
#    plt.plot([0, 0.97], [0.03, 1],    'g--', linewidth=0.8, label='$\pm$ 3%')
#    plt.xlim(0.85, 1.01)
#    plt.ylim(0.85, 1.01)
#    plt.ylabel('predicted')
#    plt.xlabel('Measured')
#    plt.title('Dose Blended Images for Portal Dosimetry')
#    plt.legend()
#    plt.savefig(f'output/Plot_egression{i}.png', bbox_inches='tight')
#    i = i+1
#print('all  ok')

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

    idx = np.random.choice(np.arange(len(p)), 205, replace=False)
    ltm1 = ltm[idx]
    mu1 = mu[idx]
    p1 = p[idx]
    y1 = y[idx]
    y21 = y2[idx]

    X_train0, X_test0, X_train2, X_test2, X_train3, X_test3, y_train, y_test, y_train2, y_test2 = train_test_split(ltm1, mu1, p1, y1, y21, test_size=0.2)
    print('X_train', X_train0.shape)
    print('X_test', X_test0.shape)
    X_train1 = X_train0.reshape(X_train0.shape[0], 70, 177, 1)
    X_test1  = X_test0.reshape(X_test0.shape[0], 70, 177, 1)
    scaler= MinMaxScaler()
    X_train2 = scaler.fit_transform(X_train2)
    X_test2 = scaler.fit_transform(X_test2)
    X_train2 = X_train2.reshape(X_train0.shape[0], 176, 1)
    X_test2  = X_test2.reshape(X_test0.shape[0], 176, 1)
    X_train3 = X_train3.reshape(X_train0.shape[0], 512, 512, 1)
    X_test3 = X_test3.reshape(X_test0.shape[0], 512, 512, 1)


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