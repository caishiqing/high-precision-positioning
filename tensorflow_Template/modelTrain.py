import scipy.io as sio                        
import h5py 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #屏蔽log 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #强制GPU运行代码

from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
# import matplotlib.pyplot as plt

from modelDesign_2 import Model_2
from modelDesign_1 import Model_1
'''
Load training data
CIR_length = 256, BS = 18;
'''

file_name1 = '../test_dataset/CIR_1T4R_100UE_2048sample.npy'
print('The current dataset is : %s'%(file_name1))
CIR = np.load(file_name1)
trainX = CIR.transpose((2,3,0,1))  #[none, 72, 2, 256]


file_name2 = '../test_dataset/POS_1T4R_100UE_2048sample.npy'
print('The current dataset is : %s'%(file_name2))
POS = np.load(file_name2)
trainY = POS.transpose((1,0))

model = Model_1(input_shape = (72, 2, 256), output_shape = 2)
model.compile(optimizer = keras.optimizers.Adam(0.002),
          loss=['mae'],
          metrics=['mae'])

print(model.summary())

callback_define = [
    keras.callbacks.ModelCheckpoint(filepath='modelSubmit_1.h5', monitor='val_loss', save_best_only=True),
]
model.fit(trainX, trainY, batch_size = 100, epochs = 20, verbose = 1, shuffle = True, validation_split = 0.1, callbacks = callback_define)

print('END')



