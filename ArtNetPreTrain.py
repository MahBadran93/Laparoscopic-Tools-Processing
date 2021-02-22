from __future__ import print_function

import numpy as np 
import os
import cv2
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import model_from_yaml

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import cv2
import glob
import itertools
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.initializers import Constant
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
'''
from skimage.morphology import disk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import jaccard_similarity_score
'''
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#...................Model............................
def pretrained(nClasses, input_height, input_width):
  img_input = Input(shape=(input_height, input_width, 3)) 

  #...................Feature Generator..............................
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
  print(FrG.shape)
    #........................................................................................................

  vgg_Base = VGG16(weights = 'imagenet',
                    include_top = False,
                    input_tensor = img_input) 
  
  conv_14 = SeparableConv2D(filters = 1024, 
                            kernel_size = (3, 3), 
                            activation = 'relu', 
                            kernel_initializer='glorot_uniform', 
                            padding="same")(vgg_Base.output)
  conv_14 = BatchNormalization()(conv_14)


  conv_15 = SeparableConv2D(filters = 1024, 
                            kernel_size = (3, 3), 
                            activation = 'relu', 
                            kernel_initializer='glorot_uniform', 
                            padding="same")(conv_14)
  conv_15 = BatchNormalization()(conv_15)

  
  deconv_1 = UpSampling2D(size = (2, 2))(conv_15)
  deconv_1 = concatenate([vgg_Base.get_layer(name="block4_pool").output,
                          deconv_1], axis=-1)
  deconv_1 = SeparableConv2D(filters = 512, 
                              kernel_size = (3, 3), 
                              activation = 'relu', 
                              kernel_initializer='glorot_uniform', 
                              padding = "same")(deconv_1)
  deconv_1 = BatchNormalization()(deconv_1)


  deconv_2 = UpSampling2D(size = (2, 2))(deconv_1)
  deconv_2 = concatenate([vgg_Base.get_layer(name="block3_pool").output,
                          deconv_2], axis=-1)
  deconv_2 = SeparableConv2D(filters = 256,
                              kernel_size = (3, 3),
                              activation = 'relu',
                              kernel_initializer='glorot_uniform',
                              padding = "same")(deconv_2)
  deconv_2 = BatchNormalization()(deconv_2)


  deconv_3 = UpSampling2D( size = (2, 2))(deconv_2)
  deconv_3 = concatenate([vgg_Base.get_layer(name="block2_pool").output,
                          deconv_3], axis=-1)
  deconv_3 = SeparableConv2D(filters = 128,
                              kernel_size = (3, 3),
                              activation = 'relu',
                              kernel_initializer='glorot_uniform',
                              padding = "same")(deconv_3)     
  deconv_3 = BatchNormalization()(deconv_3)

  
  kept = BatchNormalization()(deconv_3)

  tool = UpSampling2D(size = (2, 2))(kept)
  tool = concatenate([vgg_Base.get_layer( name="block1_pool").output, 
                      tool], axis=-1)
  
  tool = SeparableConv2D(filters = 64,
                          kernel_size = (3, 3), 
                          activation = 'relu',
                          kernel_initializer='glorot_uniform',
                          padding = "same")(tool)
  tool = BatchNormalization()(tool)

  tool = UpSampling2D(size = (2, 2))(tool)
  tool = SeparableConv2D(filters = 64, 
                          kernel_size = (3, 3), 
                          activation = 'relu',
                          kernel_initializer='glorot_uniform', 
                          padding = "same")(tool)
  tool = BatchNormalization()(tool)

  tool = SeparableConv2D(filters = nClasses,
                          kernel_size = (1, 1),
                          activation = 'relu',
                          kernel_initializer='glorot_uniform',
                          padding = "same")(tool)
  tool = BatchNormalization()(tool)

  tool = concatenate([tool, FrG], axis=-1)

  tool = Conv2D(filters = 1,
                kernel_size = 1,
                activation = 'sigmoid',
                name='tool')(tool)

  modeltool = Model(img_input,tool)
  print(modeltool.summary())

  return modeltool
  

#.....................Loss..............................
from tensorflow.keras import backend as K

def IoU(y_true, y_pred):
        
    ''' 
    The Intersection over Union (IoU) also referred to as the Jaccard index (JI),
    is essentially a method to quantify the percent overlap between the GT mask
    and prediction output. The IoU metric measures the number of pixels common 
    between the target and prediction masks divided by the total number of pixels
    present across both masks.
  
    Input Arguments: 
        y_true: True Labels of the 2D images so called ground truth (GT).
        y_pred: Predicted Labels of the 2D images so called Predicted/ segmented Mask.
        
    Output Arguments: 

        iou: The IoU between y_true and y_pred

    Author: Md. Kamrul Hasan, 
            Erasmus Scholar on Medical Imaging and Application (MAIA)
            E-mail: kamruleeekuet@gmail.com

    '''
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection) / (K.sum(y_true_f) + K.sum(y_pred_f)-intersection)


def IoU_loss(y_true, y_pred):
    return 1-IoU(y_true, y_pred)


def bce_IoU_loss(y_true, y_pred):
    return (binary_crossentropy(y_true, y_pred) + IoU_loss(y_true, y_pred))




#....................Training...........................
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#..................Data..........................
#Roboticdataset_raw = np.load('/content/drive/MyDrive/MasterThesis /Data/NPYFiles/RoboticDataTraining_Raw.npy', mmap_mode='r+')
maskTraining = np.load('./Data/NPY/maskTraining.npy/arr_0.npy').astype(np.float64)
rawTraining =  np.load('./Data/NPY/rawTraining.npy/arr_0.npy')

# Split the data (train, validation)
x_train, x_valid, y_train, y_valid = train_test_split(rawTraining, maskTraining,test_size=0.33, shuffle= True)

# init model 
model = pretrained(2,192,256)

# Compile 
model.compile(optimizer = Adadelta(learning_rate=0.01, rho=0.95, epsilon=1e-07, name="Adadelta"), loss = bce_IoU_loss, metrics = IoU)
# save the weights
checkpoint_path = "./Model/Checkpoint"
batch_size = 8
cp_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True, save_freq = 10*batch_size)
results = model.fit(x=x_train,y=y_train,validation_data=(x_valid,y_valid),batch_size=batch_size,epochs=150,callbacks=[cp_checkpoint])

# Save the model
model.save('./Model/pretrainedRobotic3.h5')