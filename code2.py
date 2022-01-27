import tensorflow as tf
import pandas as pd
import seaborn as sns
import os

#os.makedirs("output")

print('hello')

p0 = pd.read_csv('HL_Prostate_3Gy.csv') 
fig =sns.scatterplot(data=p0, x='MU', x_jitter=True, y='2_1', color='r', edgecolor='black', linewidth=0.9, alpha=0.6, label='2.7Gy').get_figure()
fig.savefig("output/image1.png")