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
#x = Dense(90, activation='relu')(x)
#x = Dense(1, activation=activation)(x)
#model1 = Model(i, x)
##Model->2
i2 = Input(shape=(176,1))
x2 = Conv1D(filters=70, kernel_size=(5), activation='relu', padding='same')(i2)
x2 = MaxPool1D(pool_size=(3))(x2)
x2 = Dropout(rate=0.1)(x2)
x2 = Flatten()(x2)
x2 = BatchNormalization()(x2)
#x = Dense(90, activation='relu')(x)
#x = Dense(1, activation='sigmoid')(x)
#model2 = Model(i, x)
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
#x = Dense(360, activation='relu')(x)
#x = Dense(90, activation='relu')(x)
#x = Dense(1, activation=activation)(x)
#model3 = Model(i, x)
#####################
#merge
merge = concatenate([x1, x2, x3])
#merge = concatenate([x2, x3])
x = Dense(360, activation='relu')(merge)
x = Dense(180, activation='relu')(x)
#x = Dense(90, activation='relu')(x)
x1 = Dense(1, activation='sigmoid')(x)
x2 = Dense(1, activation='linear')(x)
model3 = Model(inputs=[i1, i2, i3], outputs=x1)
model4 = Model(inputs=[i1, i2, i3], outputs=x2)
#model3 = Model(inputs=[i1, i3], outputs=x1)
#model4 = Model(inputs=[i1, i3], outputs=x2)


# %%
#y_cat_train = to_categorical(y_train, 2)
#y_cat_test = to_categorical(y_test, 2)

data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=10)
train_generator = data_generator.flow(X_train3, y_train,  batch_size=64)
test_generator = data_generator.flow(X_test3, y_test, shuffle=False)

#data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
#train_generator = data_generator.flow(X_train, y_train)
#test_generator = data_generator.flow(X_test, y_test, shuffle=False)
#train_generator = data_generator.flow([X_train1, X_train2], y_train)
#test_generator = data_generator.flow([X_test1, X_test2], y_test, shuffle=False)

early_stop = EarlyStopping(monitor='val_loss', patience=15)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=10, min_lr=0.00001)

models = [model3, model3, model3, model3, model3]

print('all ok')
tprs1 = []
aucs1 = []
fprs1 = []
mean_fpr = np.linspace(0, 1, 100)
i = 1
fig1, ax1 = plt.subplots()
for model in models:
    model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
    model.fit(x=[X_train1, X_train2, X_train3], y= y_train, validation_data= ([X_test1, X_test2, X_test3], y_test), epochs=200 ,verbose=0, callbacks=[early_stop, reduce_lr])
    #model.fit_generator(train_generator, validation_data= test_generator, callbacks=[early_stop], epochs=200, verbose=0)
  
    y_pred_keras = model.predict((X_test1, X_test2, X_test3)).ravel() 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_keras)
    tprs1.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs1.append(roc_auc)    

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
ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic DBPD")
ax1.legend(loc="right", bbox_to_anchor=(1.65, 0.5))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('output/drop_00.png', bbox_inches='tight')

metrics = pd.DataFrame(model.history.history)
pred = model.predict((X_test1, X_test2, X_test3))
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

print('now regression')


models = [model4, model4, model4, model4]
i = 0
for model in models:

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.fit(x=[X_train1, X_train2, X_train3], y= y_train2, validation_data= ([X_test1, X_test2, X_test3], y_test2), callbacks=[reduce_lr], epochs=200, verbose=0)
    pred2 = model4.predict((X_test1, X_test2, X_test3))
    mae = mean_absolute_error(y_test2, pred2)
    rmse = mean_squared_error(y_test2, pred2)
    print(f'MAE{i}', mae)
    print(f'RMSE{i}', rmse)
    print(f'y_test2{i}>>>','', np.array(y_test2))
    print(f'pred2>>>{i}','', np.array(pred2.ravel()))

    plt.figure(4+i)
    plt.scatter(x=y_test2, y=pred2, edgecolors='k', color='g', alpha=0.7, label='Predict')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='0%', alpha=.8)
    plt.plot([0.03, 1], [0, 0.97], 'g--', linewidth=0.8)
    plt.plot([0, 0.97], [0.03, 1],    'g--', linewidth=0.8, label='$\pm$ 3%')
    plt.xlim(0.85, 1.01)
    plt.ylim(0.85, 1.01)
    plt.ylabel('predicted')
    plt.xlabel('Measured')
    plt.title('Dose Blended Images for Portal Dosimetry')
    plt.legend()
    plt.savefig(f'output/Plot_egression{i}.png', bbox_inches='tight')
    i = i+1