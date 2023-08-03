# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models,initializers,regularizers,Model
from tensorflow.keras.layers import Reshape, Activation, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Lambda, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50

# %%

# Loading test data 
x_test = np.load('videoframes_clips_valid.npy', mmap_mode='c')

# Loading test labels
y_test = np.load('joint_3d_clips_valid.npy', mmap_mode='c')
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)




# %%
#defining the model
model = models.Sequential ([
    Lambda(lambda x: tf.divide(tf.cast(x,tf.float32),255.0)),
    
   
    
    TimeDistributed(ResNet50(input_shape=(224,224,3),include_top=False, weights='imagenet')),
    
    TimeDistributed(Dense(128)),
    
    TimeDistributed(Flatten()),
    
    LSTM(units=256, return_sequences=True),
   
    TimeDistributed(Dense(51, activation='linear')),
    Reshape((y_test.shape[1],y_test.shape[2],y_test.shape[3]))
    
])
# Loading the Model weights
model.load_weights("model_weighhts")





# %%
# Predicting the Test set results

minibatch_size = 4
test_dataset = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(minibatch_size)
)

tmp_MPJPE = []

#calculating the MPJPE
for x, y in test_dataset:
   
   
    predict = model.predict(x)
    
    MPJPE = tf.math.reduce_mean(tf.math.reduce_euclidean_norm((y - predict), axis=3)) * 1000
    tmp_MPJPE.append(MPJPE)
  
test_MPJPE = tf.reduce_mean(tmp_MPJPE)
print('MPJPE:', test_MPJPE.numpy())



# %%



