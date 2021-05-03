# Importing Dependencies
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import glob
from skimage.morphology import disk
from tensorflow.keras.utils import plot_model, to_categorical



'''
    difference = cv2.subtract(img_pathmasP[:,:,0:1], img_pathmasP[:,:,1:2])    
    result = not np.any(difference)
    if result is True:
        print("Pictures are the same")
    else:
        print("Pictures are different")
    
'''
'''
#Setting directories
cDir = os.getcwd()

TI=glob.glob(cDir+'/DataFinal/Test/Test_Positive/*.png')
TI.sort()
print(len(TI))

TM=glob.glob(cDir+'/DataFinal/Test/Test_Positive_MidLine/*.png')
TM.sort()
print(len(TM))

TMas=glob.glob(cDir+'/DataFinal/Test/Test_Positive_Tool_Mask/*.png')
TMas.sort()
print(len(TMas))

TU=glob.glob(cDir+'/DataFinal/Test/Test_Positive_EdgeLine_1/*.png')
TU.sort()
print(len(TU))

TL=glob.glob(cDir+'/DataFinal/Test/Test_Positive_EdgeLine_2/*.png')
TL.sort()
print(len(TL))

TT=glob.glob(cDir+'/DataFinal/Test/Test_Positive_TipPoint/*.png')
TT.sort()
print(len(TT))


TN=glob.glob(cDir+'/DataFinal/Test/Test_Negative/*.png')
TN.sort()
print(len(TN))



testing_data = []
XX=zip(TI, TM, TU, TL, TT, TMas, TN)
i=0
kernel2=disk(4)
for pathorigP, pathmidP, pathupP, pathlowP, pathtipP, pathmasP, pathN in XX:
    img_pathorigP = cv2.resize(cv2.imread(pathorigP,-1),(256,192))
    img_pathorigP=img_pathorigP/img_pathorigP.max()
    #print(img_pathorigP.shape)
    
    img_pathmidP = cv2.resize(cv2.imread(pathmidP,-1),(256,192))
    img_pathmidP=img_pathmidP/img_pathmidP.max()
    img_pathmidP = img_pathmidP[:,:,0:1]
    #print(img_pathmidP.shape)
    
    img_pathupP = cv2.resize(cv2.imread(pathupP,-1),(256,192))
    img_pathupP=img_pathupP/img_pathupP.max()
    img_pathupP = img_pathupP[:,:,0:1]
    #print(img_pathupP.shape)
    
    img_pathlowP = cv2.resize(cv2.imread(pathlowP,-1),(256,192))
    img_pathlowP=img_pathlowP/img_pathlowP.max()
    img_pathlowP = img_pathlowP[:,:,0:1]
    #print(img_pathlowP.shape)
    
    img_pathmasP = cv2.resize(cv2.imread(pathmasP,-1),(256,192))
    img_pathmasP=img_pathmasP/img_pathmasP.max()
    img_pathmasP = img_pathmasP[:,:,0:1]
    #print(img_pathmasP.shape) # problem hereeeeeeeeeeeeee
    #cv2.imshow('mask', img_pathmasP[:,:,0:1])
    #cv2.waitKey(20)
    
    #print(img_pathmasP.shape)
    img_pathtipP = cv2.resize(cv2.imread(pathtipP,-1),(256,192))
    img_pathtipP = cv2.dilate(img_pathtipP,kernel2,iterations = 1)
    img_pathtipP=img_pathtipP/img_pathtipP.max()
    img_pathtipP = img_pathtipP[:,:,0:1]
    #print(img_pathtipP.shape)
    
    EdgeLine = img_pathupP + img_pathlowP
    EdgeLine=EdgeLine/EdgeLine.max()
    EdgeLine = EdgeLine[:,:,0:1]

    #print(EdgeLine.shape)

    
       
    label=1
    
    testing_data.append([np.array(img_pathorigP),
                          np.array(img_pathmasP),
                          np.array(EdgeLine),
                          np.array(img_pathmidP),
                          np.array(img_pathtipP),
                          np.array(label)])

    img_pathN=cv2.resize(cv2.imread(pathN,-1),(256,192))
    img_pathN=img_pathN/img_pathN.max()
    #print(img_pathN.shape)
    
    #img_pathNDummy=img_pathN[:,:,0].reshape(192,256)
    img_pathNDummy=img_pathN[:,:,0:1]
    #print(img_pathN.shape, img_pathNDummy.shape)
    img_pathmasP=np.zeros_like(img_pathNDummy)    
    cv2.circle(img_pathmasP,(110,110), 1, 255, -1)
    img_pathmasP=img_pathmasP/img_pathmasP.max()
    #cv2.imshow('mask',img_pathmasP)
    #cv2.waitKey(20)
    #print(img_pathmasP.shape)
    
    EdgeLine=np.zeros_like(img_pathNDummy)    
    cv2.circle(EdgeLine,(110,110), 1, 255, -1)
    EdgeLine=EdgeLine/EdgeLine.max()
    
    #print(EdgeLine.shape)

    img_pathmidP=np.zeros_like(img_pathNDummy)    
    cv2.circle(img_pathmidP,(110,110), 1, 255, -1)
    img_pathmidP=img_pathmidP/img_pathmidP.max()
    #print(img_pathmidP.shape)
    
    img_pathtipP=np.zeros_like(img_pathNDummy)    
    cv2.circle(img_pathtipP,(110,110), 1, 255, -1)
    img_pathtipP=img_pathtipP/img_pathtipP.max()
    #print(img_pathtipP.shape)
    
    label=0
    
    testing_data.append([np.array(img_pathN),
                          np.array(img_pathmasP),
                          np.array(EdgeLine),
                          np.array(img_pathmidP),
                          np.array(img_pathtipP),
                          np.array(label)])

shuffle(testing_data)
'''

#print('sssssssssssssssssssss',np.array(testing_data).shape)

#np.save('./DataFinal/training_data.npy', training_data) 
#np.save('./DataFinal/testing_data.npy', testing_data)   


data = np.load('./DataFinal/testing_data.npy',allow_pickle=True)

label=to_categorical(data[20][5], num_classes=2, dtype='float32')
print(label)
'''
for i in range(data.shape[0]):
    #data[i][0] = data[i][0].reshape(-1,192,256,3)
    print(data[i][5])
'''




#for i in range(data.shape[0]):
    #cv2.imshow('data',data[i][1])
    #cv2.waitKey(50)
#print(np.unique(data[50][4]))
