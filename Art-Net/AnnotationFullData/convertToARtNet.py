import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import glob
from skimage.morphology import disk

#Setting directories Training
cDir = os.getcwd()

TI=glob.glob('/home/badran/Desktop/AnnotationFullData/TestAnnotation/PositiveImage/*.png')
TI.sort()
print(len(TI))

TM=glob.glob('/home/badran/Desktop/AnnotationFullData/TestAnnotation/MidLine/*.png')
TM.sort()
print(len(TM))

TMas=glob.glob('/home/badran/Desktop/AnnotationFullData/TestAnnotation/toolmask/*.png')
TMas.sort()
print(len(TMas))

TU=glob.glob('/home/badran/Desktop/AnnotationFullData/TestAnnotation/Edgeline1/*.png')
TU.sort()
print(len(TU))

TL=glob.glob('/home/badran/Desktop/AnnotationFullData/TestAnnotation/EdgeLine2/*.png')
TL.sort()
print(len(TL))

TT=glob.glob('/home/badran/Desktop/AnnotationFullData/TestAnnotation/Tooltip/*.png')
TT.sort()
print(len(TT))


training_data = []
XX=zip(TI, TM, TU, TL, TT, TMas)
i=0
kernel2=disk(4)
for pathorigP, pathmidP, pathupP, pathlowP, pathtipP, pathmasP in XX:
    ss = cv2.imread(pathorigP)
    img_pathorigP = cv2.resize(cv2.imread(pathorigP,-1),(256,192))
    img_pathorigP=img_pathorigP/img_pathorigP.max()
    #print(img_pathorigP.shape)
 
    img_pathmidP = cv2.resize(cv2.imread(pathmidP,-1),(256,192))
    img_pathmidP=img_pathmidP/img_pathmidP.max()
    img_pathmidP = np.expand_dims(img_pathmidP, axis=2)
    print(img_pathmidP.shape)

    img_pathupP = cv2.resize(cv2.imread(pathupP,-1),(256,192))
    img_pathupP=img_pathupP/img_pathupP.max()
    img_pathupP = np.expand_dims(img_pathupP, axis=2)
    print(img_pathupP.shape)
    
    img_pathlowP = cv2.resize(cv2.imread(pathlowP,-1),(256,192))
    img_pathlowP=img_pathlowP/img_pathlowP.max()
    img_pathlowP = np.expand_dims(img_pathlowP, axis=2)
    print(img_pathlowP.shape)
    
    img_pathmasP = cv2.resize(cv2.imread(pathmasP,-1),(256,192)) 
    img_pathmasP=img_pathmasP/img_pathmasP.max()
    img_pathmasP = img_pathmasP[:,:,0:1]
    print(img_pathmasP.shape)
    
    img_pathtipP = cv2.resize(cv2.imread(pathtipP,-1),(256,192))
    img_pathtipP = cv2.dilate(img_pathtipP,kernel2,iterations = 1)
    img_pathtipP=img_pathtipP/img_pathtipP.max()
    img_pathtipP = np.expand_dims(img_pathtipP, axis=2)
    print(img_pathtipP.shape)
    
    EdgeLine = img_pathupP + img_pathlowP
    EdgeLine=EdgeLine/EdgeLine.max()    
    #EdgeLine = np.expand_dims(EdgeLine, axis=2)
    print(EdgeLine.shape)
    #cv2.imwrite('/home/mahmoud/Desktop/image2.jpg',EdgeLine*255)
    
    #print(EdgeLine.dtype)
    #cv2.imwrite('/home/mahmoud/Desktop/image.png',np.uint8(EdgeLine))
     
    label=1


    training_data.append([np.array(img_pathorigP),
                          np.array(img_pathmasP),
                          np.array(EdgeLine),
                          np.array(img_pathmidP),
                          np.array(img_pathtipP),
                          np.array(label)])

#TN=glob.glob('/home/mahmoud/Desktop/laparoscopic-Tools-Segmentation/DataFinal/Train/Train_Negative/*.png')
#TN.sort()
#print(len(TN))