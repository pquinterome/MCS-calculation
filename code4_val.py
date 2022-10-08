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
from tensorflow.keras.layers import Input, Dense, BatchNormalization ,Conv1D, Conv2D, MaxPool2D, MaxPool1D, Flatten, Dropout, GlobalMaxPooling2D, concatenate, SimpleRNN
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
# Note

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


i1 = Input(shape=(70,177,1))
x1 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(i1)
x1 = MaxPool2D(pool_size=(2,2))(x1)
x1 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(x1)
x1 = MaxPool2D(pool_size=(2,2))(x1)                                                
x1 = Dropout(rate=0.2)(x1)
x1 = Flatten()(x1)
x1 = Dense(90, activation='relu')(x1)
x1 = Dense(1, activation='linear')(x1)
model1 = Model(i1, x1)


i2 = Input(shape=(176,1))
x2 = Conv1D(filters=70, kernel_size=(5), activation='relu', padding='same')(i2)
x2 = MaxPool1D(pool_size=(3))(x2)
x2 = Dropout(rate=0.1)(x2)
x2 = Flatten()(x2)
x2 = BatchNormalization()(x2)
x2 = Dense(90, activation='relu')(x2)
x2 = Dense(1, activation='linear')(x2)
model2 = Model(i2, x2)

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
x3 = BatchNormalization()(x3)
x3 = Dense(360, activation='relu')(x3)
x3 = Dense(90, activation='relu')(x3)
x3 = Dense(1, activation='linear')(x3)
model3 = Model(i3, x3)

roc = tf.keras.metrics.AUC(name='roc')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
early_stop = EarlyStopping(monitor='val_loss', patience=3)

model1.compile(loss="mean_squared_error", optimizer= 'adam', metrics=['mean_absolute_error'])
model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model3.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model1.fit(x= X_train1, y =y_train2, validation_data= (X_test1, y_test2), callbacks=[early_stop, reduce_lr] ,epochs=400, verbose=0)
model1.save('models/model_1_reg.h5')
model2.fit(x= X_train2, y =y_train2, validation_data= (X_test2, y_test2), callbacks=[early_stop, reduce_lr] ,epochs=400, verbose=0)
model2.save('models/model_2_reg.h5')
model3.fit(x= X_train3, y =y_train2, validation_data= (X_test3, y_test2), callbacks=[early_stop, reduce_lr] ,epochs=400, verbose=0)
model3.save('models/model_3_reg.h5')




print('X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X')
val_ltm = np.load('inputs/tlm_val.npy')
val_ltm = np.array([val_ltm[i][:177,].T for i in range(len(val_ltm))])

y3 = pd.read_csv('inputs/id_val.csv')
y3['2_2'] = y3['2_2'].fillna(y3['2_2'].mean())
y3 = y3['2_2']/100
y_test2 = y3
y= np.array([0 if x >= 0.98 else 1 for x in y3])

ltm = val_ltm
ltm = [abs(ltm[w]/ltm[w].max()) for w in range(len(ltm))]
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
mu = np.array([mu[w]/mu[w].max() for w in range(len(mu))])
p = np.load('inputs/portal_val.npy', allow_pickle=True)
p= np.array([p[w]/p[w].max() for w in range(len(p))])
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
y_test = y
print('X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X')

model1 = tf.keras.models.load_model('models/model_1_reg.h5')
model2 = tf.keras.models.load_model('models/model_2_reg.h5')
model3 = tf.keras.models.load_model('models/model_3_reg.h5')


print('all ok (:')

#metrics = pd.DataFrame(model1.history.history)
#fig = plt.figure(1)
#fig.set_size_inches(13, 5)
#plt.subplot(2,3,1)
#plt.title('Loss')
#plt.plot(metrics[['loss', 'val_loss']], label=['loss', f'val_loss'])
##plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.subplot(2,3,2)
#plt.title('Accuracy')
#plt.plot(metrics[['accuracy', 'val_accuracy']], label=['acc', 'val_acc'])
##plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.subplot(2,3,3)
#plt.title('roc_auc')
#plt.plot(metrics[['roc', 'val_roc']], label=['auc', 'val_auc'])
#plt.savefig('output/Performance_classification.png', bbox_inches='tight')

pred = model1.predict(X_test1)
mae = mean_absolute_error(y_test2, pred)
rmse = mean_squared_error(y_test2, pred)
print('MAE', mae)
print('RMSE', rmse)

pred2 = model2.predict(X_test2)
mae2 = mean_absolute_error(y_test2, pred2)
rmse2 = mean_squared_error(y_test2, pred2)
print('MAE2', mae2)
print('RMSE2', rmse2)

pred3 = model3.predict(X_test3)
mae3 = mean_absolute_error(y_test2, pred2)
rmse3 = mean_squared_error(y_test2, pred2)
print('MAE3', mae3)
print('RMSE3', rmse3)

#print('y_test2>>>','', np.array(y_test2))
#print('pred2>>>','', np.array(pred2.ravel()))

fig = plt.figure(1)
plt.scatter(x=y_test2, y=pred, edgecolors='k', color='g', alpha=0.7)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='0%', alpha=.8)
plt.plot([0.03, 1], [0, 0.97], 'g--', linewidth=0.8)
plt.plot([0, 0.97], [0.03, 1],    'g--', linewidth=0.8, label='$\pm$ 3%')
plt.xlim(0.85, 1.01)
plt.ylim(0.85, 1.01)
plt.ylabel('predicted')
plt.xlabel('Measured')
plt.title('M_1')
plt.legend()
plt.savefig('output/Plot_egression1.png', bbox_inches='tight')

fig = plt.figure(2)
plt.scatter(x=y_test2, y=pred2, edgecolors='k', color='g', alpha=0.7)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='0%', alpha=.8)
plt.plot([0.03, 1], [0, 0.97], 'g--', linewidth=0.8)
plt.plot([0, 0.97], [0.03, 1],    'g--', linewidth=0.8, label='$\pm$ 3%')
plt.xlim(0.85, 1.01)
plt.ylim(0.85, 1.01)
plt.ylabel('predicted')
plt.xlabel('Measured')
plt.title('M_2')
plt.legend()
plt.savefig('output/Plot_egression2.png', bbox_inches='tight')

fig = plt.figure(3)
plt.scatter(x=y_test2, y=pred3, edgecolors='k', color='g', alpha=0.7)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='0%', alpha=.8)
plt.plot([0.03, 1], [0, 0.97], 'g--', linewidth=0.8)
plt.plot([0, 0.97], [0.03, 1],    'g--', linewidth=0.8, label='$\pm$ 3%')
plt.xlim(0.85, 1.01)
plt.ylim(0.85, 1.01)
plt.ylabel('predicted')
plt.xlabel('Measured')
plt.title('M_3')
plt.legend()
plt.savefig('output/Plot_egression3.png', bbox_inches='tight')

#min_x = min([len(loss[i]) for i in range(len(loss))])
#rloss = [np.array([loss[j][i] for j in range(len(loss))]).mean() for i in range(min_x)]
#r_val_loss = [np.array([val_loss[j][i] for j in range(len(val_loss))]).mean() for i in range(min_x)]
#rm1 = [np.array([m1[j][i] for j in range(len(m1))]).mean() for i in range(min_x)]
#r_loss_m1 = [np.array([loss_m1[j][i] for j in range(len(loss_m1))]).mean() for i in range(min_x)]

#plt.figure(1)
#plt.title('Loss [bce]')
#plt.plot(rloss, label=['train'], color=('blue'))
#plt.plot(r_val_loss, label=['loss'], color=('orange'))
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.savefig('output/loss.png', bbox_inches='tight')

#plt.figure(2)
#plt.title('accuracy')
#plt.plot(rm1, label=['mean_acc'], color=('blue'))
#plt.plot(r_loss_m1, label=['val_acc'], color=('orange'))
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.savefig('output/acc.png', bbox_inches='tight')

#mean_tpr = np.mean(tprs1, axis=0)
#mean_tpr[-1] = 1.0
#mean_auc = auc(mean_fpr, mean_tpr)
#mean_auc = np.mean(aucs1)
#roc_auc = metrics.auc(mean_fpr, mean_tpr)
#std_auc = np.std(aucs1)
#std_tpr = np.std(tprs1, axis=0)
#tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

#plt.figure(3)
#plt.title("Receiver operating characteristic example")
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
#plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
#plt.legend(loc="right", bbox_to_anchor=(1.65, 0.5))
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.xlim(-0.05, 1.05)
#plt.ylim(-0.05, 1.05)
#plt.savefig('output/roc_auc.png', bbox_inches='tight')

#print('Mean_auc-->>', mean_auc, std_auc)

# %%
