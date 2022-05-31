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
ltm = np.concatenate((ltm, ltm[-411:]), axis=0)
p = np.concatenate((p, p[-411:]), axis=0)
mu= np.concatenate((mu, mu[-411:]), axis=0)
y = np.concatenate((y,y[-411:]), axis=0)
y2 = np.concatenate((y2,y2[-411:]), axis=0)

#print('dataset', ltm.shape)
#print('labels', y.shape)
#print('MU_cp', mu.shape)
#%%
X_train1, X_test1, X_train2, X_test2, X_train3, X_test3, y_train, y_test, y_train2, y_test2 = train_test_split(ltm, mu, p, y, y2, test_size=0.2, random_state= 35)
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
activation = 'sigmoid' 
##softmax
##models---->>>
##i = Input(shape=(70,177,1))
i = Input(shape=(512,512,1))
##1 Single layers
##Model->1
x = Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
#x = Dense(180, activation='relu')(x)
x = Dense(1, activation=activation)(x)
model1 = Model(i, x)
##Model->2
x = Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(3,3))(x)
x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
#x = Dense(180, activation='relu')(x)
x = Dense(1, activation=activation)(x)
model2 = Model(i, x)
###Model->3
x = Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(3,3))(x)
x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(360, activation='relu')(x)
x = Dense(1, activation=activation)(x)
model3 = Model(i, x)


# %%
#y_cat_train = to_categorical(y_train, 2)
#y_cat_test = to_categorical(y_test, 2)

#data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
#train_generator = data_generator.flow(X_train3, y_train,  batch_size=10)
#test_generator = data_generator.flow(X_test3, y_test, shuffle=False)

#data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
#train_generator = data_generator.flow(X_train, y_train)
#test_generator = data_generator.flow(X_test, y_test, shuffle=False)
#train_generator = data_generator.flow([X_train1, X_train2], y_train)
#test_generator = data_generator.flow([X_test1, X_test2], y_test, shuffle=False)

early_stop = EarlyStopping(monitor='val_loss', patience=15)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=10, min_lr=0.00001)

models = [model1, model2, model3]

print('all ok')

i = 1
for model in models:
    model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
    r = model.fit(x=X_train3, y= y_train, validation_data= (X_test3, y_test), epochs=200, batch_size=10 ,verbose=0, callbacks=[early_stop, reduce_lr])
    #r = model.fit(train_generator, validation_data= test_generator, callbacks=[early_stop], epochs=100, verbose=0)
    metrics = pd.DataFrame(model.history.history)
    pred = model.predict(X_test3)
    predictions = np.round(pred)
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    classes=[0,1]
    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predictions).numpy()
    
    print(f'AUC_model{i}',  roc_auc)
    print(f'Accuracy{i}',   accuracy_score(y_test, predictions))
    print(f'precision{i}',  precision_score(y_test, predictions))
    print(f'recall{i}',     recall_score(y_test, predictions))
    print(f'f1{i}',         f1_score(y_test, predictions))#

    plt.figure(i*i)
    plt.title('Loss')
    plt.plot(metrics[['loss', 'val_loss']], label=[f'loss{i}', f'val_loss{i}'])
    #plt.ylim(-0.1, 2)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'output/loss{i}.png', bbox_inches='tight')#

    plt.figure(i*i+1)
    plt.title('Accuracy')
    plt.plot(metrics[['accuracy', 'val_accuracy']], label=[f'acc{i}', f'val_acc{i}'])
    #plt.ylim(0.4, 1.1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'output/acc{i}.png', bbox_inches='tight')##

    plt.figure(15)
    plt.title("Receiver operating characteristic example")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.2)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label=(f'ROC{i}', roc_auc))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.xlim(-0.05, 1.05)
    #plt.ylim(-0.05, 1.05)
    plt.savefig('output/auc.png', bbox_inches='tight')
    
    i = i+1
