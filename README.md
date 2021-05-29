# Recognize Painted Digit

## Content:

* Demo
* Overview
* The model
* Paint screen
* Tools

## Demo:
   Video link : [youtu.be](https://youtu.be/nWiEWk9BGg4)

## Overview : 
This is an image detection project based on digit recognition using convolution neural network and MNIST dataset.

## The model: 
I used in this model the convolution neural network with 3 layers of Conv2D nd 1 hidden layer the dense layer and  output layer, also I used the optimizer adam and loss sparse_categorical_crossentropy which is appropriate for detecting categories also I used dropout to avoid model overfitting and callbacks for early stopping as I set epochs to 30 which will allow the model to continue learning until the accuracy stopped improving, below is the model summary

_________________________________________________
Layer (type)                 Output Shape              Param  
=================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 128)         73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 1, 1, 128)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 1, 1, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 128)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               16512     
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================
Total params: 110,474
Trainable params: 110,474
Non-trainable params: 0
________________________________________________________________

### Note: contain saved model weights file (Saved-Recognized-Painted-Digit-Model.h5)

## Paint screen:
I used OpenCV and numpy to display black screen for drawing the digit using click mouse events and then applying the model for prediction.

## Tools: 
Python, Keras, OpenCV and MNIST dataset
