# %%
from cProfile import label
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
#os.makedirs("output")
print('hello')
# %%
#p0 = pd.read_csv('HL_Prostate_3Gy.csv') 
#fig =sns.scatterplot(data=p0, x='MU', x_jitter=True, y='2_1', color='r', edgecolor='black', linewidth=0.9, alpha=0.6, label='2.7Gy').get_figure()
#fig.savefig("output/image1.png")
ltm1 = np.load('tlm_3Gy_2arc_HL.npy')
ltm1 = np.array([ltm1[i][:112,] for i in range(len(ltm1))])
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
X_train, X_test, y_train, y_test = train_test_split(ltm, y, test_size=0.2) #random_state=1
X_train = X_train.reshape(437,112,177,1)
X_test = X_test.reshape(110,112,177,1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# %%
data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
train_generator = data_generator.flow(X_train, y_train, batch_size=3)

i = Input(shape=(112,177,1))
x = Conv2D(filters=64, kernel_size=(3,1), activation='relu')(i)
x = Conv2D(filters=64, kernel_size=(1,3), activation='relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
#x = GlobalMaxPooling2D()(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(60, activation='relu')(x)
x = Dense(38, activation='relu')(x)
x = Dense(15, activation='relu')(x)
x = Dense(1, activation='linear')(x)
model1 = Model(i, x)

model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
early_stop = EarlyStopping(monitor='val_loss', patience=3)

r = model1.fit(train_generator, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)


scores = model1.evaluate(X_test, y_test, verbose=0)
print(model1.metrics_names[0], 'of',  {scores[0]})

metrics = pd.DataFrame(model1.history.history)

plt.figure(1) 
plt.subplot(211)
plt.title('Loss [rmse]')
plt.plot(metrics[['loss', 'val_loss']], label=['train', 'loss'])
plt.legend()
plt.subplot(212)
plt.title('Mean Absolute Error')
plt.plot(metrics[['mean_absolute_error', 'val_mean_absolute_error']], label=['mean_abs_error', 'val_mean_abs_error' ])
plt.legend()
plt.tight_layout()
plt.savefig('output/learning_curves.png', bbox_inches='tight')

X_test[0].shape
test_generator = data_generator.flow(X_test, y_test, batch_size=3)
pred = model1.predict(test_generator)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)
print('MAE---->>>>', mae)
print('RMSE--->>>>', rmse)


plt.figure(2) 
plt.scatter(x=y_test, y=pred, edgecolors='k', color='g', alpha=0.7)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.ylabel('predicted')
plt.xlabel('Measured')
plt.ylim(0.9, 1.01)
plt.xlim(0.9, 1.01)
plt.savefig('output/predictions.png', bbox_inches='tight')

corr = spearmanr(pred, y_test)
print('Spearman Correlation', corr)
# %%
G = y
mu=[0 if x >= 0.975 else 1 for x in G]
X1 = ltm
X2 = X1.reshape(X1.shape[0],X1.shape[1],X1.shape[2],1)
y2 = np.array(mu)
seed =18
np.random.seed(seed)
tprs1 = []
aucs1 = []
mean_fpr = np.linspace(0, 1, 100)
fold_no = 1
i1 = 1
plt.figure(3)
fig, ax = plt.subplots()
kfold = StratifiedKFold(n_splits=5, shuffle=True) 
for train, test in kfold.split(X2, y2):
    
    print(f'fold_no {fold_no}')
    # create model
    i = Input(shape=(112,177,1))
    x = Conv2D(filters=64, kernel_size=(3,1), activation='relu')(i)
    x = Conv2D(filters=64, kernel_size=(1,3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    #x = GlobalMaxPooling2D()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(600, activation='relu')(x)
    x = Dense(300, activation='relu')(x)
    x = Dense(150, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(i, x)
    auc1 = tf.keras.metrics.AUC()
    model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=[auc1])
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(x=X2[train], y= y2[train], validation_data=(X2[test], y2[test]),
                epochs=600,verbose=0, callbacks=[early_stop]) #batch=size=5
    #metrics = pd.DataFrame(model.history.history)
    #metrics.plot()  
# evaluate the model  
    y_pred_keras = model.predict(X2[test]).ravel()
 
    fpr, tpr, thresholds = roc_curve(y2[test], y_pred_keras)
    tprs1.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    #roc_auc5 = metrics.auc(fpr, tpr)
    aucs1.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i1, roc_auc))
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
# %%
X1 = ltm
X = X1.reshape(X1.shape[0],X1.shape[1],X1.shape[2],1)
y = y
seed =18
np.random.seed(seed)
mae = []
rmse = []
lossE = []
m1=[]
loss_m1=[]
val_lossE=[]
fold_no = 1
plt.figure(4)
fig, ax = plt.subplots()
kfold = KFold(n_splits=3, shuffle=True) #, random_state=seed)

for train, test in kfold.split(X, y):
    
    print(f'fold_no {fold_no}')
    
    data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    train_generator = data_generator.flow(X[train], y[train], batch_size=3)
  # create model
    i = Input(shape=(112,177,1))
    x = Conv2D(filters=64, kernel_size=(3,1), activation='relu')(i)
    x = Conv2D(filters=64, kernel_size=(1,3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(2,2), activation='relu')(x)
    #x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(600, activation='relu')(x)
    x = Dense(300, activation='relu')(x)
    x = Dense(150, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    model = Model(i, x)    
## compile model     
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    batch_size = 3
    
    r = model.fit(train_generator, epochs=200, validation_data=(X[test], y[test]), callbacks=[early_stop], verbose=0)

    metrics = pd.DataFrame(model.history.history)
    
    plt.subplot(211)
    plt.title('Loss [rmse]')
    plt.plot(metrics['loss'], color=('blue'), alpha=0.1, label='_nolegend_')
    plt.plot(metrics['val_loss'], color=('orange'), alpha=0.1, label='_nolegend_')
    plt.subplot(212)
    plt.title('MAE')
    plt.plot(metrics['mean_absolute_error'], color=('blue'), alpha=0.1, label='_nolegend_')
    plt.plot(metrics['val_mean_absolute_error'], color=('orange'), alpha=0.1, label='_nolegend_')
       
# evaluate the model  
    test_generator = data_generator.flow(X[test], y[test], batch_size=3)
    pred = model.predict(test_generator)
    mae_i = mean_absolute_error(y[test], pred)
    rmse_i = mean_squared_error(y[test], pred, squared=True)

    mae.append(mae_i)
    rmse.append(rmse_i)

    lossE.append(np.array(metrics['loss']))
    val_lossE.append(np.array(metrics['val_loss']))
    m1.append(np.array(metrics['mean_absolute_error']))
    loss_m1.append(np.array(metrics['val_mean_absolute_error']))

    fold_no = fold_no + 1

m = metrics['val_mean_absolute_error']
mean = metrics['val_mean_absolute_error'].mean()
std = metrics['val_mean_absolute_error'].std()
m2 = metrics['val_loss']
mean2 = metrics['val_loss'].mean()
std2 = metrics['val_loss'].std()
print('CV_mae=', mean, std)
print('CV_rmse=', mean2, std2)

rloss = [np.array([lossE[j][i] for j in range(len(lossE))]).mean() for i in range(len(lossE[0]))]
r_val_loss = [np.array([val_lossE[j][i] for j in range(len(val_lossE))]).mean() for i in range(len(val_lossE[0]))]
rm1 = [np.array([m1[j][i] for j in range(len(m1))]).mean() for i in range(len(m1[0]))]
r_loss_m1 = [np.array([loss_m1[j][i] for j in range(len(loss_m1))]).mean() for i in range(len(loss_m1[0]))]

plt.subplot(211)
plt.title('Loss [rmse]')
plt.plot(rloss, label=['train'], color=('blue'))
plt.plot(r_val_loss, label=['loss'], color=('orange'))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplot(212)
plt.title('MAE')
plt.plot(rm1, label=['mean_absolute_error'], color=('blue'))
plt.plot(r_loss_m1, label=['val_mean_absolute_error'], color=('orange'))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
#plt.savefig('output/train_curves_CV.png', bbox_inches='tight')


















# %%
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
    #x = Dense(38, activation='relu')(x)
    #x = Dense(15, activation='relu')(x)
    x= Dense(1, activation='linear')(x)
    model = Model(i, x)    
    #compile model     
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    
    r = model.fit(train_generator, epochs=600, validation_data=test_generator, callbacks=[early_stop], verbose=0)
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
    #test_generator = data_generator.flow(X[test], y[test], batch_size=3)
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
#plt.savefig('output/loss.png', bbox_inches='tight')
plt.figure(2)
plt.title('MAE')
plt.plot(rm1, label=['mean_absolute_error'], color=('blue'))
plt.plot(r_loss_m1, label=['val_mean_absolute_error'], color=('orange'))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.tight_layout()
#plt.savefig('output/mae.png', bbox_inches='tight')
plt.figure(3)
plt.scatter(x=y[test], y=pred, edgecolors='k', color='g', alpha=0.7)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.ylabel('predicted')
plt.xlabel('Measured')
plt.ylim(0.9, 1.01)
plt.xlim(0.9, 1.01)






# %%
# Cross Validation
X = ltm.reshape(ltm.shape[0],ltm.shape[1],ltm.shape[2],1)
G = y
mu=[0 if x >= 0.975 else 1 for x in G]
y = np.array(mu)
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
    data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)#, zoom_range=[0.7,1.0]) #, shear_range=0.1)
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
    x = Dense(600, activation='relu')(x)
    x = Dense(300, activation='relu')(x)
    x = Dense(150, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
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