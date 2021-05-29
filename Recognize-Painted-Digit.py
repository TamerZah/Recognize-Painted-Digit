# -*- coding: utf-8 -*-
"""
Created on Thu May 27 21:21:31 2021

@author: lenovo
"""

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

dataset = tf.keras.datasets.mnist

# Spliting data
(X_train, y_train), (X_test, y_test) = dataset.load_data()

# normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0


# CNN input shape: (Batch size, height, width, color channel)
# ANN input shape: (Batch size, features)
# Features = Width * Height
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

print(X_train.shape)
print(X_test.shape)


# Import packages
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU




model = Sequential()

# 32 Number of filters it should increase within the depth of the model
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

# After last conv layer and maxpooling and before first fully connected layer add flatten
model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation = 'softmax'))

# After the model is created we will need the gradient descent to compute the cost "J"
# by using optimization algorithm as adam which is stochastic gradient descen algorithm 
# or rmsprop or other so to do this use compile function as below
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.summary()

callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=2, verbose=1), 
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, verbose=1)]

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 64, epochs = 30, callbacks=callbacks)

# model.save('Saved-Recognized-Painted-Digit-Model.h5')

# model = tf.keras.models.load_model('Saved-Recognized-Painted-Digit-Model.h5')



# Paint prediction

run = False
ix, iy = -1, -1
follow = 25
img = np.zeros((512, 512, 1))


# draw Fuction

def draw(event, x, y, flag, params):
    global run, ix, iy, img, follow
    
    if event == cv2.EVENT_LBUTTONDOWN:
        run = True;
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if run == True:
            cv2.circle(img, (x, y), 10, (255,255,255), -1)
            
    elif event == cv2.EVENT_LBUTTONUP:
        run = False
        cv2.circle(img, (x, y), 10, (255,255,255), -1)
        drawimg = cv2.resize(img, (28, 28))
        drawimg = drawimg.reshape(1, 28, 28, 1)
        result = np.argmax(model.predict(drawimg))
        result = 'No: {}'.format(result)
        cv2.putText(img, result, (25, follow), 0, 1, (255,255,255), 3)
        follow += 25
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((512, 512, 1))
        follow = 25
        
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

while True:
    
    cv2.imshow('image', img)
    
    if cv2.waitKey(1) == ord('x'):
        break
    
cv2.destroyAllWindows()
    
