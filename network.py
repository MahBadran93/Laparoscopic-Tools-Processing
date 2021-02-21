# -*- coding: utf-8 -*-
from __future__ import division
import torch
#import torch 
#import torch.nn as nn
#import torch.nn.functional as F 
#from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from dataload import DataLoader


#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# run using gpu or cpu 

with tf.device('/cpu:0'):

    #.............................Data Set...........................................
    data = DataLoader()
    masks_class,masks_inst, raw = data.loadDataRigid('./Data') 

    masks_inst = keras.applications.vgg16.preprocess_input(masks_inst)
    masks_class = keras.applications.vgg16.preprocess_input(masks_class)
    raw = keras.applications.vgg16.preprocess_input(raw)


    # .....................Input shape that will enter the network.....................
    inputEndo = Input(shape=(480, 640, 3))


    # ...................Feature Extraction Generator Block............................    
    nClasses = 2                          
    FrG = SeparableConv2D(filters = 64,
                          kernel_size = (3, 3), 
                          activation = 'relu',
                          kernel_initializer='glorot_uniform',
                          padding="same")(img_input)
    FrG = BatchNormalization()(FrG)
    
    FrG = SeparableConv2D(filters = 256,
                          kernel_size = (3, 3),
                          activation = 'relu', 
                          kernel_initializer='glorot_uniform', 
                          padding="same")(FrG)
    FrG = BatchNormalization()(FrG)
        
    FrG = SeparableConv2D(filters = 64,
                          kernel_size = (3, 3), 
                          activation = 'relu',
                          kernel_initializer='glorot_uniform',
                          padding="same")(FrG)
    FrG = BatchNormalization()(FrG)
    
    FrG = SeparableConv2D(filters = nClasses,
                          kernel_size = (3, 3), 
                          activation = 'relu',
                          kernel_initializer='glorot_uniform',
                          padding="same")(FrG)
    FrG = BatchNormalization()(FrG)


    #...............................................................................

    #........................ Encoeder vgg16 with dense output layer................
    #.......................... Detection part......................................

    # Pretrined model, trained on 'imagenet' dataset. THE ENCODER    
    vgg_Base = keras.applications.vgg16.VGG16(weights = 'imagenet',
                        include_top = False,
                        input_tensor = inputEndo)

    

    # convert functional model to sequential model. 
    vgg_sequential = keras.Sequential()
    for layer in vgg_Base.layers:
        vgg_sequential.add(layer)

    # Freeze the layers in case of any future training because of fine tuning 
    for layer in vgg_sequential.layers:
        layer.trainable = False

    # add flatten lyer to vgg16 to feed its output to Dense layer 
    vgg_sequential.add(keras.layers.Flatten())
    vgg_sequential.add(keras.layers.Dense(2,activation='softmax'))    

    #........................................................................................
     
   

    #vgg_Base.predict(raw)
    #plt.imshow(raw[0])
    #plt.show()

    
    print(vgg_sequential.summary())      


    with tf.device('/gpu:0'):
        pass
        #vgg_Base.fit(raw, masks_class, batch_size=5,epochs=50)
    

# Check device to train 


#vgg_Base.fit(x=raw, y=masks_inst, batch_size=20,epochs=100)


