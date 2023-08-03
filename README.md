# Dynamic-CNN-LSTM-FC-Model-for-Body-Pose-Estimation
In this programming assignment, our goal is to implement a deep dynamic model to perform the task of human body pose estimation. The dynamic model consists of
CNN layers to absorb the spatial information of data frames, LSTM layers to represent the temporal coherence between consecutive data frames and MLP layers to 
perform the final regression task for predicting 17 body joint co-ordinates. The dataset that we use is a subset of the Human3.6M dataset, where the input is 
video sequence of RGB frames and output is 17 body joint co-ordinates, each having a 3D value represented as a cartesian 3D point. The dataset that we have
been provided has 5964 video sequence for training and 1368 video sequence for validation. Each video sequence consists of 8 frames and each frame has a dimension
of 224x224x3 where the last dimension represents 3 RGB channels. The network we use has a pre-trained ResNet-50 architecture at the beginning, then a dense layer 
to downsample the number of connections, a flattening layer, then LSTM layer and finally output layer for regression. The network has been trained by minimizing 
the Mean Squared Error between the predicted output and the ground truth joint co-ordinate values
