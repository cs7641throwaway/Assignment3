import pandas as pd
import numpy as np
import os
import gzip

# Modified from jontay to handle new datasets

# fashionmnist

path = '../fashionmnist/'
kind = 'train'

labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

with gzip.open(labels_path, 'rb') as lbpath:
	labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

with gzip.open(images_path, 'rb') as imgpath:
	images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

print(images.shape)
print(labels.shape)

images_df = pd.DataFrame(data=images).astype(float)
labels_df = pd.DataFrame(data=labels)
labels_df.columns = ['Class']
fmnist = pd.concat([images_df, labels_df],1)

print("Images head: ")
print(images_df.head())
print("Labels head: ")
print(labels_df.head())
print("Dataframe head: ")
print(fmnist.head())

fmnist.to_hdf('datasets_full.hdf','fmnist',complib='blosc',complevel=9)

# chess

path = '../chess/'
kind = 'kr-vs-kp'

# All in one
data_labels_path = os.path.join(path, '%s.data' % kind)

chess = pd.read_csv(data_labels_path,header=None)

chess.columns = ['bkblk','bknwy','bkon8','bkona','bkspr','bkxbq','bkxcr','bkxwp','blxwp','bxqsq','cntxt','dsopp','dwipd',
                 'hdchk','katri','mulch','qxmsq','r2ar8','reskd','reskr','rimmx','rkxwp','rxmsq','simpl','skach','skewr',
                 'skrxp','spcop','stlmt','thrsk','wkcti','wkna8','wknck','wkovl','wkpos','wtoeg', 'win']

chess = pd.get_dummies(chess)
chess['win'] = chess['win_won']
chess = chess.drop(['win_won', 'win_nowin'], axis=1)

print("Chess head: ")
print(chess.head())

chess.to_hdf('datasets_full.hdf','chess',complib='blosc',complevel=9)
