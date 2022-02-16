#%%
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from numpy import interp
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.metrics import plot_roc_curve, auc, precision_score, recall_score, f1_score, roc_curve
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import shapiro
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from numpy import interp
# %%
ltm1 = np.load('tlm_3Gy_1arc_HL.npy')
ltm2 = np.load('tlm_3Gy_2arc_HL.npy')
ltm2 = np.array([ltm2[i][:112,] for i in range(len(ltm2))])
ltm = np.concatenate((ltm1, ltm2), axis=0)
ltm = np.array([ltm[i] for i in range(len(ltm))])
ltm = np.array([(ltm[i]-ltm[i].mean())/ltm[i].std() for i in range(len(ltm))])
ltm = np.array([ltm[i]/ltm[i].max() for i in range(len(ltm))])
ltm_H =np.array([(ltm[i]+1)/2 for i in range(len(ltm))]) 
print('Halcyon', ltm_H.shape)
ltm1 = np.load('tlm_3Gytb_1arc.npy')
ltm2 = np.load('tlm_27Gytb_1arc.npy')
ltm3 = np.load('tlm_2Gy_1arc.npy')
ltm = np.concatenate((ltm1, ltm2, ltm3), axis=0)
ltm = np.array([ltm[i].T for i in range(len(ltm))])
ltm = np.array([(ltm[i]-ltm[i].mean())/ltm[i].std() for i in range(len(ltm))])
ltm = np.array([ltm[i]/ltm[i].max() for i in range(len(ltm))])
ltm_T =np.array([(ltm[i]+1)/2 for i in range(len(ltm))])
ltm_T = np.array([ltm_T[i][:112,] for i in range(len(ltm_T))])
print('TrueBeam', ltm_T.shape)
ltm = np.concatenate((ltm_H, ltm_T), axis=0)
ltm= ltm.reshape(822, 112, 177, 1)
y = np.load('y.npy')
print('dataset', ltm.shape)
print('labels', y.shape)
print('X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X-X')
#%%
#models---->>>

#1 Single layers
#Model->1
i = Input(shape=(112,177,1))
x = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(180, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)
model1 = Model(i, x)
#Model->2
i = Input(shape=(112,177,1))
x = Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(180, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)
model2 = Model(i, x)
#Model->3
i = Input(shape=(112,177,1))
x = Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(180, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)
model3 = Model(i, x)
#Model->4
i = Input(shape=(112,177,1))
x = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(180, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)
model4 = Model(i, x)
#Model->5
i = Input(shape=(112,177,1))
x = Conv2D(filters=128, kernel_size=(2,2), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(180, activation='relu')(x)
x = Dense(180, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)
model5 = Model(i, x)



# %%
G = y
mu=[0 if x >= 0.98 else 1 for x in G]
y = np.array(mu)

X_train, X_test, y_train, y_test = train_test_split(ltm, y, random_state = 1, test_size=0.2) #random_state=1


#data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, zoom_range=[0.7,1.0], 
#                                    shear_range=0.1, validation_split=0.2, featurewise_center=True, 
#                                    featurewise_std_normalization=True)
#train_generator = data_generator.flow(X_train, y_train)
#test_generator = data_generator.flow(X_test, y_test, shuffle=False)
early_stop = EarlyStopping(monitor='val_loss', patience=10)
#auc1 = tf.keras.metrics.AUC()

models = [model1, model2, model3, model4]
i = 1
for model in models:
    model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])
    r = model.fit(x=X_train, y= y_train, validation_data= (X_test, y_test), epochs=500, verbose=0, callbacks=[])
    metrics = pd.DataFrame(model.history.history)
    pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)  

    print(f'AUC_model{i}', roc_auc)

    plt.figure(i*i)
    plt.title('Loss [rmse]')
    plt.plot(metrics[['loss', 'val_loss']], label=[f'loss{i}', f'val_loss{i}'])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'output/loss{i}.png', bbox_inches='tight')

    plt.figure(i*i+1)
    plt.title('Mean Absolute Error')
    plt.plot(metrics[['accuracy', 'val_accuracy']], label=[f'acc{i}', f'val_acc{i}'])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'output/acc{i}.png', bbox_inches='tight')

    plt.figure(15)
    plt.title("Receiver operating characteristic example")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.2)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label=(f'ROC{i}', roc_auc))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.savefig('output/auc.png', bbox_inches='tight')

    i = i+1




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
