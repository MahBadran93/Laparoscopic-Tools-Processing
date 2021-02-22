import tensorflow as tf
#from tensorflow.keras import backend as K


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
    return (tf.keras.losses.binary_crossentropy(y_true, y_pred) + IoU_loss(y_true, y_pred))



#..............Load NPY Data....................................

import numpy as np

# load testing data
gtTesting = np.load('./Data/NPY/gtTesting.npy/arr_0.npy').astype(np.float32)
rawTesting =  np.load('./Data/NPY/rawTesting.npy/arr_0.npy').astype(np.float32)
gtTesting = gtTesting[:4495]


#print(maskTraining.shape, rawTraining.shape, gtTesting.shape, rawTesting.shape)

#for i in range(Roboticdataset_mask.shape[0]):
  #Roboticdataset_mask[i][Roboticdataset_mask[i] != 0] = 1

#...........................Test Model...............................
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('./Models/pretrainedRobotic2.h5', custom_objects={'bce_IoU_loss': bce_IoU_loss, 'IoU': IoU})
model2 = load_model('./Models/IoU.h5', custom_objects={'bce_IoU_loss': bce_IoU_loss, 'IoU': IoU})

image = np.expand_dims(rawTesting[0], 0)

pred = model.predict(image)
pred2 = model2.predict(image)

image = pred.astype(np.float64)
image2 = pred2.astype(np.float64)
#image = pred/pred.max()
#image2 = pred2/pred2.max()

#image = np.where(pred > 0, 1, 0)
#image2 = np.where(pred2 > 0, 1, 0)


print(image.shape, image2.shape, image.dtype, image2.dtype,np.unique(image), np.unique(image2))

# image = rawTraining
#results = model.evaluate(rawTesting,gtTesting) 
#print(results)
# image = np.expand_dims(image,axis=0)
# pred = model.predict(image)




import matplotlib.pyplot as plt
plt.imshow(np.squeeze(image))
#plt.show()
'''
plt.imshow(np.squeeze(gtTesting[4000]))
plt.show()
'''