#%%
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pyparsing import alphas
import seaborn as sns
from sklearn.utils import shuffle
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
aucs1 = []
tprs1 = []
mean_fpr = np.linspace(0, 1, 100)
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
i1=1
kfold = KFold(n_splits=5, shuffle=True) #, random_state=seed)

plt.figure(1)
for train, test in kfold.split(X, y):
    print(f'fold_no {fold_no}')
    #Data Generator
    batch_size = 16
    data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, zoom_range=[0.7,1.0], shear_range=0.1)
    train_generator = data_generator.flow(X[train], y[train])
    test_generator = data_generator.flow(X[test], y[test], shuffle=False)
    #create model
    i = Input(shape=(112,177,1))
    #x = GlobalMaxPooling2D()(i)  
    x = Conv2D(filters=32, kernel_size=(3,1), activation='elu')(i)
    x = Conv2D(filters=32, kernel_size=(1,3), activation='elu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    #x = GlobalMaxPooling2D()(x)  
    #x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x= Dense(1, activation='sigmoid')(x)
    model = Model(i, x)   
    #compile model     
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    r = model.fit(train_generator, epochs=600, validation_data=test_generator, callbacks=[early_stop], verbose=0)
    metrics = pd.DataFrame(model.history.history)
    
    plt.figure(1)
    plt.title('Loss [rmse]')
    plt.plot(metrics['loss'], color=('blue'), alpha=0.1, label='_nolegend_')
    plt.plot(metrics['val_loss'], color=('orange'), alpha=0.1, label='_nolegend_')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.figure(2)
    plt.title('Accuracy')
    plt.plot(metrics['accuracy'], color=('blue'), alpha=0.1, label='_nolegend_')
    plt.plot(metrics['val_accuracy'], color=('orange'), alpha=0.1, label='_nolegend_')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
       
# evaluate the model  
    #test_generator = data_generator.flow(X[test], y[test], batch_size=3)
    pred = model.predict(X[test]).ravel()

    fpr, tpr, thresholds = roc_curve(y2[test], pred)
    tprs1.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    #roc_auc5 = metrics.auc(fpr, tpr)
    aucs1.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i1, roc_auc))

    lossE.append(np.array(metrics['loss']))
    val_lossE.append(np.array(metrics['val_loss']))
    m1.append(np.array(metrics['accuracy']))
    loss_m1.append(np.array(metrics['val_accuracy']))

    i1= i1+1
    fold_no = fold_no + 1




ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs1, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
#mean_auc = np.mean(aucs1)
#roc_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs1)

ax.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs1, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="right", bbox_to_anchor=(1.65, 0.5))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('output/roc_auc.png', bbox_inches='tight')
print('Mean_auc-->>', mean_auc, std_auc)

rloss = [np.array([lossE[j][i] for j in range(len(lossE))]).mean() for i in range(len(lossE[0]))]
r_val_loss = [np.array([val_lossE[j][i] for j in range(len(val_lossE))]).mean() for i in range(len(val_lossE[0]))]
rm1 = [np.array([m1[j][i] for j in range(len(m1))]).mean() for i in range(len(m1[0]))]
r_loss_m1 = [np.array([loss_m1[j][i] for j in range(len(loss_m1))]).mean() for i in range(len(loss_m1[0]))]

plt.figure(2)
plt.title('Loss [rmse]')
plt.plot(rloss, label=['train'], color=('blue'))
plt.plot(r_val_loss, label=['loss'], color=('orange'))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.savefig('output/loss.png', bbox_inches='tight')
plt.figure(3)
plt.title('Acc')
plt.plot(rm1, label=['mean_absolute_error'], color=('blue'))
plt.plot(r_loss_m1, label=['val_mean_absolute_error'], color=('orange'))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.tight_layout()
plt.savefig('output/acc.png', bbox_inches='tight')


# %%
