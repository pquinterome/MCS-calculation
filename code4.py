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
from sklearn.metrics import plot_roc_curve, auc, precision_score, recall_score, f1_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, MaxPool2D, MaxPool1D, Flatten, Dropout, GlobalMaxPooling2D, concatenate, SimpleRNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from scipy.stats import shapiro
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from numpy import interp
# %%
ltm_T = np.load('inputs/ltm_T.npy')
ltm_H = np.load('inputs/ltm_H.npy')
ltm = np.concatenate((ltm_H, ltm_T), axis=0)
y = np.load('y.npy')
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
mu = np.load('mu_cp.npy')
ltm = np.concatenate((ltm, ltm[-411:]), axis=0)
y = np.concatenate((y,y[-411:]), axis=0)
mu= np.concatenate((mu, mu[-411:]), axis=0)
#print('dataset', ltm.shape)
#print('labels', y.shape)
#print('MU_cp', mu.shape)
#%%
X_train1, X_test1, X_train2, X_test2, y_train, y_test = train_test_split(ltm, mu, y, test_size=0.2, random_state= 35)
#print('X_train', X_train1.shape)
#print('X_test', X_test1.shape)
X_train1 = X_train1.reshape(986, 70, 177, 1)
X_test1  = X_test1.reshape(247, 70, 177, 1)
scaler= MinMaxScaler()
X_train2 = scaler.fit_transform(X_train2)
X_test2 = scaler.fit_transform(X_test2)
X_train2 = X_train2.reshape(986, 176, 1)
X_test2  = X_test2.reshape(247, 176, 1)
print('X_train1', X_train1.shape)
print('X_test1', X_test1.shape)
print('X_train2', X_train2.shape)
print('X_test2', X_test2.shape)
print('y_test', y_test.shape)
print('X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X')
#%%
#'X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X'
i = Input(shape=(70,177,1))
x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
#x = Dense(360, activation='relu')(x)
#x = Dense(2, activation='softmax')(x)
x = Dense(1, activation='sigmoid')(x)
model1 = Model(i, x)
#model1.summary()
data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
train_generator = data_generator.flow(X_train1, y_train, batch_size=10)
test_generator = data_generator.flow(X_test1, y_test, shuffle=False)
roc = tf.keras.metrics.AUC(name='roc')
#adam= tf.keras.optimizers.Adam(learning_rate=0.0005, name='adam')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.01)
early_stop = EarlyStopping(monitor='val_loss', patience=3)
model1.compile(loss="binary_crossentropy", optimizer= 'adam', metrics=['accuracy', roc])
model1.fit(train_generator, validation_data= test_generator, callbacks=[early_stop] ,epochs=400, verbose=0)
metrics = pd.DataFrame(model1.history.history)

fig = plt.figure(1)
fig.set_size_inches(13, 5)
plt.subplot(2,3,1)
plt.title('Loss')
plt.plot(metrics[['loss', 'val_loss']], label=['loss', f'val_loss'])
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplot(2,3,2)
plt.title('Accuracy')
plt.plot(metrics[['accuracy', 'val_accuracy']], label=['acc', 'val_acc'])
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplot(2,3,3)
plt.title('roc_auc')
plt.plot(metrics[['roc', 'val_roc']], label=['auc', 'val_auc'])
plt.savefig('output/Performance.png', bbox_inches='tight')
pred = model1.predict(X_test1)
predictions = np.round(pred)
print(classification_report(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
fpr1, tpr1, thresholds = roc_curve(y_test, pred)
roc_auc = auc(fpr1, tpr1)
print('roc_auc', roc_auc)
print('accuracy', accuracy)
plt.figure(2)
plt.plot(fpr1, tpr1, lw=2, alpha=0.3, label=('ROC{LTM}', "{:.2f}".format(roc_auc)))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.2)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.savefig('output/AUC_ltm.png', bbox_inches='tight')
#'X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X'
i = Input(shape=(176, 1))
x = Conv1D(filters=128, kernel_size=(9), activation='relu', padding='same')(i)
x = Conv1D(filters=32, kernel_size=(5), activation='relu', padding='same')(x)
x = MaxPool1D(pool_size=(5))(x)
x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)
model2 = Model(i, x)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
model2.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy', roc])
model2.fit(x= X_train2, y =y_train, validation_data= (X_test2, y_test), callbacks=[reduce_lr], epochs=400, verbose=0)
metrics = pd.DataFrame(model2.history.history)
fig = plt.figure(3)
fig.set_size_inches(13, 5)
plt.subplot(2,3,1)
plt.title('Loss [rmse]')
plt.plot(metrics[['loss']], label=['loss'])
plt.plot(metrics[['val_loss']], label=['val_loss'])
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend()
plt.subplot(2,3,2)
plt.title('Accuracy')
plt.plot(metrics[['accuracy', 'val_accuracy']], label=['acc', 'val_acc'])
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplot(2,3,3)
plt.title('roc_auc')
plt.plot(metrics[['roc']], label=['auc'])
plt.plot(metrics[['val_roc']], label=['val_auc'])
plt.legend()
plt.savefig('output/Performance_MU.png', bbox_inches='tight')
pred = model2.predict(X_test2)
predictions = np.round(pred)
print(classification_report(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
fpr2, tpr2, thresholds = roc_curve(y_test, pred)
roc_auc_mu = auc(fpr2, tpr2)
plt.figure(4)
plt.plot(fpr2, tpr2, lw=2, alpha=0.3, label=('ROC_MU_cp',  "{:.2f}".format(roc_auc_mu)))
plt.plot(fpr1, tpr1, lw=2, alpha=0.3, label=('ROC_LTM',    "{:.2f}".format(roc_auc)))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.2)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.savefig('output/AUC_mu.png', bbox_inches='tight')
##############################################
print('START cross validation')
##############################################

#%%
seed =18
np.random.seed(seed)
tprs1 = []
aucs1 = []
fprs1 = []
mean_fpr = np.linspace(0, 1, 100)
i = 1
fig, ax = plt.subplots()
kfold = StratifiedKFold(n_splits=5, shuffle=True) #, random_state=seed)
# for i, (train, test) in enumerate(cv.split(X_13 , target)):
X = mu.reshape(1233, 176,1)
for train, test in kfold.split(X, y):
    #!rm -rf ./logs/
    
#-  create model
    i1 = Input(shape=(176, 1))
    x = Conv1D(filters=128, kernel_size=(9), activation='relu', padding='same')(i1)
    x = Conv1D(filters=32, kernel_size=(5), activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=(5))(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    model2 = Model(i1, x)
#-  compile model    
    roc = tf.keras.metrics.AUC(name='roc')
    #adam= tf.keras.optimizers.Adam(learning_rate=0.0005, name='adam')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.01)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model2.compile(loss="binary_crossentropy", optimizer= 'adam', metrics=['accuracy', roc])
    model2.fit(x=X[train], y= y[train], validation_data=(X[test], y[test]) ,epochs=600, batch_size=5, verbose=0, callbacks=[early_stop, reduce_lr])
    #metrics = pd.DataFrame(model.history.history)
    #metrics.plot()  
#-  evaluate the model  
    y_pred_keras = model2.predict(X[test]).ravel()
    fpr, tpr, thresholds = roc_curve(y[test], y_pred_keras)
    #fprs1.append(fpr)
    tprs1.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    #roc_auc5 = metrics.auc(fpr, tpr)
    aucs1.append(roc_auc)
    #plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i= i+1

#mean_tpr = np.mean
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs1, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs1)
#mean_auc = auc(mean_fpr, mean_tpr)
#roc_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs1)

ax.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs1, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic MU")
ax.legend(loc="right", bbox_to_anchor=(1.65, 0.5))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('output/five_AUC_MU.png', bbox_inches='tight')
##############################################
print('MU model done')
##############################################
seed =18
np.random.seed(seed)
tprs1 = []
aucs1 = []
fprs1 = []
mean_fpr = np.linspace(0, 1, 100)
i = 1
fig1, ax1 = plt.subplots()
kfold = StratifiedKFold(n_splits=5, shuffle=True) #, random_state=seed)
# for i, (train, test) in enumerate(cv.split(X_13 , target)):
X = ltm.reshape(1233, 70, 177,1)
for train, test in kfold.split(X, y):
    #!rm -rf ./logs/
    data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    train_generator = data_generator.flow(X[train], y[train], batch_size=5)
    test_generator = data_generator.flow(X[test], y[test], shuffle=False)
#-  create model
    i1 = Input(shape=(70,177,1))
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(i1)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    model1 = Model(i1, x)
    #model1.summary()
##- compile model    
    roc = tf.keras.metrics.AUC(name='roc')
    #adam= tf.keras.optimizers.Adam(learning_rate=0.0005, name='adam')
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.01)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model1.compile(loss="binary_crossentropy", optimizer= 'adam', metrics=['accuracy', roc])
    model1.fit(train_generator, validation_data= test_generator ,epochs=600, batch_size=5, verbose=0, callbacks=[early_stop, reduce_lr])
    #metrics = pd.DataFrame(model.history.history)
    #metrics.plot()  
#-  evaluate the model  
    y_pred_keras = model1.predict(X[test]).ravel() 
    fpr, tpr, thresholds = roc_curve(y[test], y_pred_keras)
    #fprs1.append(fpr)
    tprs1.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    #roc_auc5 = metrics.auc(fpr, tpr)
    aucs1.append(roc_auc)
    #plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i= i+1
ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs1, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs1)
std_auc = np.std(aucs1)
ax1.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs1, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic LTM")
ax1.legend(loc="right", bbox_to_anchor=(1.65, 0.5))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('output/five_AUC_LTM.png', bbox_inches='tight')
##############################################
print('LTM model done')
##############################################