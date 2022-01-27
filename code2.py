
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
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

#os.makedirs("output")

print('hello')

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

data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
train_generator = data_generator.flow(X_train, y_train, batch_size=3)

i = Input(shape=(112,177,1))
x = Conv2D(filters=64, kernel_size=(1,3), activation='relu')(i)
x = Conv2D(filters=64, kernel_size=(3,2), activation='relu')(x)
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

metrics = pd.DataFrame(model1.history.history)

plt.figure(1) 
plt.subplot(311)
plt.title('Loss [rmse]')
plt.plot(metrics[['loss', 'val_loss']], label=['train', 'loss'])
plt.legend()
plt.subplot(312)
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
plt.ylabel('predicted')
plt.xlabel('Measured')
plt.savefig('output/predictions.png', bbox_inches='tight')

corr = spearmanr(pred, y_test)
print('Spearman Correlation', corr)