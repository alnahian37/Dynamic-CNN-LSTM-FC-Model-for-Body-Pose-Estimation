# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models,initializers,regularizers
from tensorflow.keras.layers import Reshape, Activation, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Lambda, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50

# %%
# Loading training data 

x_train = np.load('../input/program4data/data_prog4Spring22/videoframes_clips_train.npy', mmap_mode='c')

# Loading test data 

x_test = np.load('../input/program4data/data_prog4Spring22/videoframes_clips_valid.npy', mmap_mode='c')

# Loading train labels
y_train = np.load('../input/program4data/data_prog4Spring22/joint_3d_clips_train.npy', mmap_mode='c')


# Loading test labels

y_test = np.load('../input/program4data/data_prog4Spring22/joint_3d_clips_valid.npy', mmap_mode='c')

print('Data Loaded Successfully')



# %%
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# %%
# Model Hyperparameters

minibatch_size = 4
epoch = 20
learn_rate = 0.0005

# Computing mpjpe

def mpjpe(y_true, y_pred):
    val = tf.math.multiply(1000.0,tf.reduce_mean(tf.norm(y_pred-y_true, ord='euclidean', axis=3)))

    return val

# Defining a Model

model = models.Sequential ([
    Lambda(lambda x: tf.divide(tf.cast(x,tf.float32),255.0)),      
    TimeDistributed(ResNet50(input_shape=(224,224,3),include_top=False, weights='imagenet')),
    
    TimeDistributed(Dense(128)),
    
    TimeDistributed(Flatten()),
    
    LSTM(units=256, return_sequences=True),
   

    TimeDistributed(Dense(51, activation='linear')),
    Reshape((y_train.shape[1],y_train.shape[2],y_train.shape[3]))
    
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate), loss='mean_squared_error', metrics=['mse',mpjpe])
dependencies = {
    'mpjpe': mpjpe
}
        
history = model.fit(x_train, y_train, epochs=epoch, batch_size = minibatch_size, verbose=1, validation_data=(x_test, y_test))






# %%
model.summary()

# %%
# Saving the Model

model.save_weights("model_weights")


# Saving Loss and MPJPE by epochs

np.save('trainhistory.npy',history.history)


# %%
# Plotting Loss and mpjpe by epochs

all_history=np.load('trainhistory.npy',allow_pickle='TRUE').item()

train_loss = all_history["loss"]
test_loss = all_history["val_loss"]
train_mpjpe = all_history["mpjpe"]
test_mpjpe = all_history["val_mpjpe"]

# Train Loss by Epochs

plt.plot(train_loss, label ="Train Loss")
plt.plot(test_loss, label ="Test Loss")
plt.legend()
plt.xlabel('Total number of epochs')
plt.ylabel('Loss')
plt.show()

# mpjpe by Epochs

plt.plot(train_mpjpe, label ="Train mpjpe")
plt.plot(test_mpjpe, label ="Test mpjpe")
plt.legend()
plt.xlabel('Total number of epochs')
plt.ylabel('mpjpe')
plt.show()



# %%
#Model Performance on Test Data
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



