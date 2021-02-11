# -*- coding: utf-8 -*-
from __future__ import division

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

# run using gpu or cpu 
with tf.device('/cpu:0'):

    inputEndo = Input(shape=(480, 640, 3))



    vgg_Base = keras.applications.vgg16.VGG16(weights = 'imagenet',
                        include_top = False,
                        input_tensor = inputEndo) 
        
    data = DataLoader()
    masks_class,masks_inst, raw = data.loadDataRigid('./Data') 

    masks_inst = keras.applications.vgg16.preprocess_input(masks_inst)
    masks_class = keras.applications.vgg16.preprocess_input(masks_class)
    raw = keras.applications.vgg16.preprocess_input(raw)

    #plt.imshow(raw[0])
    #plt.show()

    vgg_Base.compile(optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-6) ,
                loss=keras.losses.categorical_crossentropy, 
                metrics = 'accuracy')
    #print(vgg_Base.summary())      
            
    model = keras.Sequential(vgg_Base)
    model.add(Dense(2,activation="relu"))
    model.pop()
    print(model.summary())

    with tf.device('/gpu:0'):
        pass
        #vgg_Base.fit(raw, masks_class, batch_size=5,epochs=50)
    

# Check device to train 


#vgg_Base.fit(x=raw, y=masks_inst, batch_size=20,epochs=100)



'''
def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """ 
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks


# blocks  = parse_cfg('./yolov3.cfg')

# for index, x in enumerate(blocks[1:]):
#     module = nn.Sequential()
    
#     print(type(module))

def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
'''
