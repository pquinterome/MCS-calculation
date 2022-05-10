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
#%%
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
#'X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X'
#i = Input(shape=(70,177,1))
i = Input(shape=(512, 512, 1))
x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(90, activation='relu')(x)
x1 = Dense(1, activation='sigmoid')(x)
x2 = Dense(1, activation='linear')(x)
model1 = Model(i, x1)
model2 = Model(i, x2)

#model1.summary()
data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
train_generator = data_generator.flow(X_train3, y_train, batch_size=10)
test_generator = data_generator.flow(X_test3, y_test, shuffle=False)

roc = tf.keras.metrics.AUC(name='roc')
#adam= tf.keras.optimizers.Adam(learning_rate=0.0005, name='adam')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.01)
early_stop = EarlyStopping(monitor='val_loss', patience=3)

model1.compile(loss="binary_crossentropy", optimizer= 'adam', metrics=['accuracy', roc])
model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

roc = tf.keras.metrics.AUC(name='roc')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model1.fit(x= X_train3, y =y_train, validation_data= (X_test3, y_test), callbacks=[ early_stop, reduce_lr], epochs=400, verbose=0)
model2.fit(x= X_train3, y =y_train2, validation_data= (X_test3, y_test2), callbacks=[reduce_lr] ,epochs=300, verbose=0)

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
plt.savefig('output/Performance_classification.png', bbox_inches='tight')
pred = model1.predict(X_test3)
predictions = np.round(pred)
print('LTM_Model Classification Report')
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

fig = plt.figure(3)
metrics2 = pd.DataFrame(model2.history.history)
plt.subplot(211)
plt.title('Loss [rmse]')
plt.plot(metrics2[['loss', 'val_loss']], label=['train (mean_squared_error)', 'loss'])
plt.legend()
plt.subplot(212)
plt.title('Mean Absolute Error')
plt.plot(metrics2[['mean_absolute_error', 'val_mean_absolute_error']], label=['mean_abs_error', 'val_mean_abs_error' ])
plt.legend()
plt.savefig('output/Performance_regression.png', bbox_inches='tight')

pred2 = model2.predict(X_test3)
mae = mean_absolute_error(y_test2, pred2)
rmse = mean_squared_error(y_test2, pred2)
print('MAE', mae)
print('RMSE', rmse)

print('y_test2>>>','', np.array(y_test2))
print('pred2>>>','', np.array(pred2.ravel()))

fig = plt.figure(4)
plt.scatter(x=y_test2, y=pred2, edgecolors='k', color='g', alpha=0.7)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.xlim(0.85, 1.01)
plt.ylim(0.85, 1.01)
plt.ylabel('predicted')
plt.xlabel('Measured')
plt.savefig('output/Plot_egression.png', bbox_inches='tight')
