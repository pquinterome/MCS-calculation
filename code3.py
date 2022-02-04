#%%
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pyparsing import alphas
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import shapiro
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import plot_roc_curve, auc, precision_score, recall_score, f1_score, roc_curve
from numpy import interp
# %%
ltm1 = np.load('tlm_3Gy_2arc_HL.npy')
ltm1 = np.array([ltm1[i][:112,] for i in range(len(ltm1))])     #-->To consider just 112 leaves
ltm2 = np.load('tlm_3Gy_1arc_HL.npy')
ltm = np.concatenate((ltm1, ltm2), axis=0)
y1 = pd.read_csv('HL_3Gy_2ARC.csv')
y2 = pd.read_csv('HL_3Gy_1ARC.csv')
y1['2_1'] = y1['2_1'].fillna(y1['2_1'].mean())
y2['2_1'] = y2['2_1'].fillna(y2['2_1'].mean())
y1 = np.array(y1['2_1'])
y2 = np.array(y2['2_1'])
y = np.concatenate((y1,y2), axis=0)
y = y.reshape(547)
print ('Input size', ltm.shape)
print('Output size', len(y))
print('X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X')
# %%
# Cross Validation
X = ltm.reshape(ltm.shape[0],ltm.shape[1],ltm.shape[2],1)
y= y

spear=[]
mae = []
rmse = []
lossE = []
m1=[]
loss_m1=[]
val_lossE=[]
predict=[]
yeti=[]
fold_no = 1
fig, ax = plt.subplots()
kfold = KFold(n_splits=5, shuffle=True) #, random_state=seed)


for train, test in kfold.split(X, y):
    print(f'fold_no {fold_no}')
    #Data Generator
    data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, zoom_range=[0.7,1.0], shear_range=0.1)
    train_generator = data_generator.flow(X[train], y[train], batch_size=3)
    #create model
    i = Input(shape=(112,177,1))
    x = GlobalMaxPooling2D()(i)  
    x = Conv2D(filters=32, kernel_size=(3,1), activation='relu')(i)
    x = Conv2D(filters=32, kernel_size=(1,3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    #x = GlobalMaxPooling2D()(x)  
 

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    #x = Dense(38, activation='relu')(x)
    #x = Dense(15, activation='relu')(x)
    x= Dense(1, activation='linear')(x)
    model = Model(i, x)    
    #compile model     
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    batch_size = 3
    r = model.fit(train_generator, epochs=600, validation_data=(X[test], y[test]), callbacks=[early_stop], verbose=0)
    metrics = pd.DataFrame(model.history.history)
    
    plt.figure(1)
    plt.title('Loss [rmse]')
    plt.plot(metrics['loss'], color=('blue'), alpha=0.1, label='_nolegend_')
    plt.plot(metrics['val_loss'], color=('orange'), alpha=0.1, label='_nolegend_')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.figure(2)
    plt.title('MAE')
    plt.plot(metrics['mean_absolute_error'], color=('blue'), alpha=0.1, label='_nolegend_')
    plt.plot(metrics['val_mean_absolute_error'], color=('orange'), alpha=0.1, label='_nolegend_')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
       
# evaluate the model  
    test_generator = data_generator.flow(X[test], y[test], batch_size=3)
    pred = model.predict(test_generator)

    mae_i = mean_absolute_error(y[test], pred)
    rmse_i = mean_squared_error(y[test], pred, squared=False)
    mae.append(mae_i)
    rmse.append(rmse_i)
    lossE.append(np.array(metrics['loss']))
    val_lossE.append(np.array(metrics['val_loss']))
    m1.append(np.array(metrics['mean_absolute_error']))
    loss_m1.append(np.array(metrics['val_mean_absolute_error']))

    corr = spearmanr(pred, y[test])
    spear.append(corr)

    plt.figure(3)
    plt.scatter(x=y[test], y=pred, edgecolors='k', color='g', alpha=0.1, label='_nolegend_')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.1)
    plt.ylabel('predicted')
    plt.xlabel('Measured')
    plt.ylim(0.9, 1.01)
    plt.xlim(0.9, 1.01)
    fold_no = fold_no + 1
    #i2 = i2 + 0.2

m = metrics['val_mean_absolute_error']
mean = metrics['val_mean_absolute_error'].mean()
std = metrics['val_mean_absolute_error'].std()
m2 = metrics['val_loss']
mean2 = metrics['val_loss'].mean()
std2 = metrics['val_loss'].std()
print('MAE---->>>>',    mae)
print('RMSE--->>>>',    rmse)
print('CV_mae=', mean, std)
print('CV_rmse=', mean2, std2)
print(spear)
min_x = min([len(lossE[i]) for i in range(len(lossE))])
rloss = [np.array([lossE[j][i] for j in range(len(lossE))]).mean() for i in range(min_x)]
r_val_loss = [np.array([val_lossE[j][i] for j in range(len(val_lossE))]).mean() for i in range(min_x)]
rm1 = [np.array([m1[j][i] for j in range(len(m1))]).mean() for i in range(min_x)]
r_loss_m1 = [np.array([loss_m1[j][i] for j in range(len(loss_m1))]).mean() for i in range(min_x)]
plt.figure(1)
plt.title('Loss [rmse]')
plt.plot(rloss, label=['train'], color=('blue'))
plt.plot(r_val_loss, label=['loss'], color=('orange'))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('output/loss.png', bbox_inches='tight')
plt.figure(2)
plt.title('MAE')
plt.plot(rm1, label=['mean_absolute_error'], color=('blue'))
plt.plot(r_loss_m1, label=['val_mean_absolute_error'], color=('orange'))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.tight_layout()
plt.savefig('output/mae.png', bbox_inches='tight')
plt.figure(3)
plt.scatter(x=y[test], y=pred, edgecolors='k', color='g', alpha=0.7)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.ylabel('predicted')
plt.xlabel('Measured')
plt.ylim(0.9, 1.01)
plt.xlim(0.9, 1.01)
plt.savefig('output/regression.png', bbox_inches='tight')
# %%
