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
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, MaxPool2D, Flatten, Dropout, GlobalMaxPooling2D, concatenate, SimpleRNN
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

#print('dataset', ltm.shape)
#print('labels', y.shape)
#print('MU_cp', mu.shape)
#%%
X_train1, X_test1, X_train2, X_test2, X_train3, X_test3, y_train, y_test = train_test_split(ltm, mu, p, y, test_size=0.2, random_state= 35)
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
print('X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X')
#%%
activation = 'sigmoid' 
##softmax
##models---->>>
##i = Input(shape=(70,177,1))
#i = Input(shape=(512,512,1))
##1 Single layers
##Model->1
#x = Conv2D(filters=128, kernel_size=(9,9), activation='relu', padding='same')(i)
#x = MaxPool2D(pool_size=(2,2))(x)
#x = Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='same')(x)
#x = MaxPool2D(pool_size=(2,2))(x)
#x = Flatten()(x)
#x = Dense(1, activation=activation)(x)
#model1 = Model(i, x)
##Model->2
#x = Conv2D(filters=64, kernel_size=(7,7), activation='relu', padding='same')(i)
#x = MaxPool2D(pool_size=(2,2))(x)
#x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(x)
#x = MaxPool2D(pool_size=(2,2))(x)
#x = Flatten()(x)
#x = Dense(90, activation='relu')(x)
#x = Dense(1, activation=activation)(x)
#model2 = Model(i, x)
##Model->3
#x = Conv2D(filters=128, kernel_size=(7,7), activation='relu', padding='same')(i)
#x = MaxPool2D(pool_size=(3,3))(x)
#x = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
#x = MaxPool2D(pool_size=(2,2))(x)
#x = Flatten()(x)
#x = Dense(90, activation='relu')(x)
#x = Dense(1, activation=activation)(x)
#model3 = Model(i, x)


# %%
#y_cat_train = to_categorical(y_train, 2)
#y_cat_test = to_categorical(y_test, 2)

data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
train_generator = data_generator.flow(X_train3, y_train,  batch_size=10)
test_generator = data_generator.flow(X_test3, y_test, shuffle=False)

#data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
#train_generator = data_generator.flow(X_train, y_train)
#test_generator = data_generator.flow(X_test, y_test, shuffle=False)
#train_generator = data_generator.flow([X_train1, X_train2], y_train)
#test_generator = data_generator.flow([X_test1, X_test2], y_test, shuffle=False)

early_stop = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=10, min_lr=0.00001)

#models = [model1, model2, model3]

#print('all ok')

#i = 1
#for model in models:
#    model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
#    #categorical_crossentropy
#    #binary_crossentropy
#    r = model.fit(x=X_train3, y= y_train, validation_data= (X_test3, y_test), epochs=100, batch_size=10 ,verbose=0, callbacks=[early_stop, reduce_lr])
#    #r = model.fit(train_generator, validation_data= test_generator, callbacks=[early_stop], epochs=100, verbose=0)
#    metrics = pd.DataFrame(model.history.history)
#    pred = model.predict(X_test3)
#    predictions = np.round(pred)
#    fpr, tpr, thresholds = roc_curve(y_test, pred)
#    roc_auc = auc(fpr, tpr)
#    classes=[0,1]
#    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predictions).numpy()
#    
#    print(f'AUC_model{i}',  roc_auc)
#    print(f'Accuracy{i}',   accuracy_score(y_test, predictions))
#    print(f'precision{i}',  precision_score(y_test, predictions))
#    print(f'recall{i}',     recall_score(y_test, predictions))
#    print(f'f1{i}',         f1_score(y_test, predictions))#
#
#    plt.figure(i*i)
#    plt.title('Loss')
#    plt.plot(metrics[['loss', 'val_loss']], label=[f'loss{i}', f'val_loss{i}'])
#    plt.ylim(-0.1, 2)
#    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    plt.savefig(f'output/loss{i}.png', bbox_inches='tight')#
#
#    plt.figure(i*i+1)
#    plt.title('Accuracy')
#    plt.plot(metrics[['accuracy', 'val_accuracy']], label=[f'acc{i}', f'val_acc{i}'])
#    plt.ylim(0.4, 1.1)
#    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    plt.savefig(f'output/acc{i}.png', bbox_inches='tight')#
#
#    plt.figure(15)
#    plt.title("Receiver operating characteristic example")
#    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.2)
#    plt.plot(fpr, tpr, lw=2, alpha=0.3, label=(f'ROC{i}', roc_auc))
#    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    plt.xlim(-0.05, 1.05)
#    plt.ylim(-0.05, 1.05)
#    plt.savefig('output/auc.png', bbox_inches='tight')
#
#    i = i+1
#########################
seed =18
np.random.seed(seed)
tprs1 = []
aucs1 = []
fprs1 = []
mean_fpr = np.linspace(0, 1, 100)
i = 1
fig1, ax1 = plt.subplots()
kfold = StratifiedKFold(n_splits=5, shuffle=True) #, random_state=seed)
#for i, (train, test) in enumerate(cv.split(X_13 , target)):
X = ltm.reshape(1231, 70, 177,1)
#X = p.reshape(1231, 512, 512, 1)
for train, test in kfold.split(X, y):
    #!rm -rf ./logs/
    data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    train_generator = data_generator.flow(X[train], y[train], batch_size=5)
    test_generator = data_generator.flow(X[test], y[test], shuffle=False)
    #-  create model
    i1 = Input(shape=(70,177,1))
    #1 Single layers
    # Model->1
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(i1)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(90, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model1 = Model(i1, x)
    #model1.summary()
#- compile model    
    roc = tf.keras.metrics.AUC(name='roc')
    #adam= tf.keras.optimizers.Adam(learning_rate=0.0005, name='adam')
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.01)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model1.compile(loss="binary_crossentropy", optimizer= 'adam', metrics=['accuracy', roc])
    model1.fit(x=X_train1, y= y_train, validation_data= (X_test1, y_test), epochs=100, batch_size=10 ,verbose=0, callbacks=[early_stop, reduce_lr])
    #model1.fit(train_generator, validation_data= test_generator ,epochs=600, batch_size=5, verbose=0, callbacks=[ reduce_lr])
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
plt.savefig('output/AUC_model1.png', bbox_inches='tight')
####################



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
