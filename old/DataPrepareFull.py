# Importing Dependencies
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import glob
from skimage.morphology import disk

#Setting directories
cDir = os.getcwd()

TI=glob.glob(cDir+'/DataFinal/Train/Train_Positive/*.png')
TI.sort()
print(len(TI))

TM=glob.glob(cDir+'/DataFinal/Train/Train_Positive_MidLine/*.png')
TM.sort()
print(len(TM))

TMas=glob.glob(cDir+'/DataFinal/Train/Train_Positive_Tool_Mask/*.png')
TMas.sort()
print(len(TMas))

TU=glob.glob(cDir+'/DataFinal/Train/Train_Positive_EdgeLine_1/*.png')
TU.sort()
print(len(TU))

TL=glob.glob(cDir+'/DataFinal/Train/Train_Positive_EdgeLine_2/*.png')
TL.sort()
print(len(TL))

TT=glob.glob(cDir+'/DataFinal/Train/Train_Positive_TipPoint/*.png')
TT.sort()
print(len(TT))


TN=glob.glob(cDir+'/DataFinal/TrainNeg/*.png')
TN.sort()
print(len(TN))



training_data = []
XX=zip(TI, TM, TU, TL, TT, TMas, TN)
print(XX[0])

i=0
kernel2=disk(4)
for pathorigP, pathmidP, pathupP, pathlowP, pathtipP, pathmasP, pathN in XX:
    
    img_pathorigP = cv2.resize(cv2.imread(pathorigP,-1),(256,192))
    img_pathorigP=img_pathorigP/img_pathorigP.max()
#     print(img_pathorigP.shape)
    
    img_pathmidP = cv2.resize(cv2.imread(pathmidP,-1),(256,192))
    img_pathmidP=img_pathmidP/img_pathmidP.max()
#     print(img_pathmidP.shape)
    
    img_pathupP = cv2.resize(cv2.imread(pathupP,-1),(256,192))
    img_pathupP=img_pathupP/img_pathupP.max()
#     print(img_pathupP.shape)
    
    img_pathlowP = cv2.resize(cv2.imread(pathlowP,-1),(256,192))
    img_pathlowP=img_pathlowP/img_pathlowP.max()
#     print(img_pathlowP.shape)
    
    img_pathmasP = cv2.resize(cv2.imread(pathmasP,-1),(256,192))
    img_pathmasP=img_pathmasP/img_pathmasP.max()
#     print(img_pathmasP.shape)
    
    img_pathtipP = cv2.resize(cv2.imread(pathtipP,-1),(256,192))
    img_pathtipP = cv2.dilate(img_pathtipP,kernel2,iterations = 1)
    img_pathtipP=img_pathtipP/img_pathtipP.max()
#     print(img_pathtipP.shape)
    
    EdgeLine = img_pathupP + img_pathlowP
    EdgeLine=EdgeLine/EdgeLine.max()
#     print(EdgeLine.shape)

       
    label=1
    
    training_data.append([np.array(img_pathorigP),
                          np.array(img_pathmasP),
                          np.array(EdgeLine),
                          np.array(img_pathmidP),
                          np.array(img_pathtipP),
                          np.array(label)])
    
    img_pathN=cv2.resize(cv2.imread(pathN,-1),(256,192))
    img_pathN=img_pathN/img_pathN.max()
#     print(img_pathN.shape)
    
    img_pathNDummy=img_pathN[:,:,0].reshape(192,256)
    
    img_pathmasP=np.zeros_like(img_pathNDummy)    
    cv2.circle(img_pathmasP,(110,110), 1, 255, -1)
    img_pathmasP=img_pathmasP/img_pathmasP.max()
#     print(img_pathmasP.shape)
    
    EdgeLine=np.zeros_like(img_pathNDummy)    
    cv2.circle(EdgeLine,(110,110), 1, 255, -1)
    EdgeLine=EdgeLine/EdgeLine.max()
    
#     print(EdgeLine.shape)

    img_pathmidP=np.zeros_like(img_pathNDummy)    
    cv2.circle(img_pathmidP,(110,110), 1, 255, -1)
    img_pathmidP=img_pathmidP/img_pathmidP.max()
#     print(img_pathmidP.shape)
    
    img_pathtipP=np.zeros_like(img_pathNDummy)    
    cv2.circle(img_pathtipP,(110,110), 1, 255, -1)
    img_pathtipP=img_pathtipP/img_pathtipP.max()
#     print(img_pathtipP.shape)
    
    label=0
    
    training_data.append([np.array(img_pathN),
                          np.array(img_pathmasP),
                          np.array(EdgeLine),
                          np.array(img_pathmidP),
                          np.array(img_pathtipP),
                          np.array(label)])
shuffle(training_data)
np.save('training_data.npy', training_data)    


data = np.load(cDir+'/training_data.npy')



img = np.array([i[0] for i in data]).reshape(-1,192,256,3)
mask = np.array([i[1] for i in data]).reshape(-1,192,256,1)
edge = np.array([i[2] for i in data]).reshape(-1,192,256,1)
mid  = np.array([i[3] for i in data]).reshape(-1,192,256,1)
tip = np.array([i[4] for i in data]).reshape(-1,192,256,1)
label = np.array([i[5] for i in data])

ii=0

for i in range(len(img[:,1,1,1])):
    im=img[i,:,:,:].reshape(192,256,3)
    
    immid=tip[i,:,:,:].reshape(192,256, 1)
    immid=cv2.merge((immid,immid,immid))
    immid[:,:,1]=0
    
    a=label[i]
#     a=np.argmax(lab,axis=0)
    
    mm=cv2.addWeighted(immid,0.5,im,0.7,0)
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5,50)
    fontScale              = 1
    fontColor              = (0,0,255)
    lineType               = 2
   
    cv2.putText(mm,str(a), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    cv2.imwrite(cDir+'/checkTrain/'+str(ii)+'_tip.png' , 255*mm )
    ii=ii+1