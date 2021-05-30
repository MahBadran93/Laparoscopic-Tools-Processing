import cv2
import numpy as np
import glob
import os

pathFromSuperviselyMasks = '/home/badran/Desktop/AnnotationFullData/Art-Net Tool Mask/ds0/masks_machine/*.png'
pathToSave = '/home/badran/Desktop/AnnotationFullData/TestAnnotation/toolmask/'

rotateSupervisely =glob.glob(pathFromSuperviselyMasks)

for path in rotateSupervisely:
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    image = cv2.imread(path)
    image = image * 255
    #..... here images are the same type annd shpe as kamruls data before feed to network.... 
    writePath = pathToSave + filename +'.png'
    cv2.imwrite(writePath, image)
