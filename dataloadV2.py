#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:41:37 2021

@author: mahmoud
"""

from skimage.transform import resize 
from sklearn.preprocessing import normalize 
import matplotlib.pylab as plt
import os 
from os import walk
import cv2
import numpy as np

class DataLoader():
    
    def __init__(self):
        self.imageWidth = 0 
        self.imageHeigh = 0 
        self.numChannel = 0 
        self.raws = []
        self.masks_class = [] 
        self.masks_inst = []
        self.operation1_list = []
        self.operation1_listFull = []
        self.listOfRawImages = []
    
    def resizeThreshImages(self,inputImage, width = 192, height = 256):
        gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
        image = resize(gray,(width,height))
        thresh = 0
        #im_bw = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
        image = np.array(image)
        image = np.expand_dims(image , axis = 2)
        return image
    
    def normalize(self,inputImage): 
        normalize(inputImage) 
    
    # return numpy array of rigid segmentation data provided by EncoVis (raws and instrument masks and class masks ) 
    # raw shape : (160, 480, 640, 3)
    # class masks shape : (160, 480, 640, 3)
    # instrument masks shape : (160 , 480, 640, 3)
    
    def loadDataRigid(self,path):
        raw_ImageList = []
        mask_inst_ImageList = []
        mask_class_ImageList = []
            
        for (root, dirs, files) in walk(path):

            # sort file names by number, alphapets,...
            dirs.sort()
            files.sort()
            
            if 'Masks' in root:
                for file in files:    
                    if 'instrument' in file:
                        path = os.path.join(root, file)
                        image = cv2.imread(path) 
                        mask_inst_ImageList.append(image)
                        #cv2.imshow('masks', image)
                        #cv2.waitKey(20)
                        
                    if 'class' in file:
                        path = os.path.join(root, file)
                        image = cv2.imread(path) 
                        mask_class_ImageList.append(image)
                        #cv2.imshow('masks', image)
                        #cv2.waitKey(20)
                
            if 'Raw' in root:
                for file in files:
                    path = os.path.join(root, file)
                    image = cv2.imread(path) 
                    raw_ImageList.append(image)
                    #cv2.imshow('Raw', image)
                    #cv2.waitKey(50)

        self.raws = np.array(raw_ImageList)
        self.masks_class = np.array(mask_class_ImageList)
        self.masks_inst = np.array(mask_inst_ImageList)
        
        return self.masks_class, self.masks_inst , self.raws
    
    

    
    def loadDataRobotics(self,path, dataset1=False):
        for (root, dirs, files) in walk(path):  
            if 'Dataset1' in root:
                for file in files:
                    path = os.path.join(root, file)
                    pathToFolderRaw = './Data/Segmentation_Robotic_Training/Training/Dataset1/Raw'
                    pathToFolderLeft = './Data/Segmentation_Robotic_Training/Training/Dataset1/Left'
                    pathToFolderRight = './Data/Segmentation_Robotic_Training/Training/Dataset1/Right'

                    if 'Video' in path:
                        print('vid')
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            cv2.imwrite(os.path.join(pathToFolderRaw, "rawframe%d.jpg" % i), image)
                            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            #self.resizeThreshImages(image)
                            #self.listOfRawImages.append(image)                            
                            i+=1
                        
                        cap.release()
                        cv2.destroyAllWindows()
                    
                    if 'Left' in path:
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            cv2.imwrite(os.path.join(pathToFolderLeft, "leftmaskframe%d.jpg" % i), image)

                            #image = self.resizeThreshImages(image)
                            #self.operation1_list.append(image)

                            i+=1
                        
                        cap.release()
                        cv2.destroyAllWindows()
                        

                    if 'Right' in path:
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            cv2.imwrite(os.path.join(pathToFolderRight, "rightmaskframe%d.jpg" % i), image)
                            #image = self.resizeThreshImages(image)
                            #self.operation1_list.append(image)

                            i+=1
                        cap.release()
                        cv2.destroyAllWindows()

                        #merge left nd right tools in one mask
                        #num = int(len(self.operation1_list)/2)
                        #for i in range(num):
                            #self.operation1_listFull.append(self.operation1_list[i] + self.operation1_list[i + num])

            if 'Dataset2' in root:
                for file in files:
                    path = os.path.join(root, file)
                    pathToFolderRaw = './Data/Segmentation_Robotic_Training/Training/Dataset2/Raw'
                    pathToFolderMask = './Data/Segmentation_Robotic_Training/Training/Dataset2/Mask'

                    if 'Video' in path:
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            cv2.imwrite(os.path.join(pathToFolderRaw, "rawframe%d.jpg" % i), image)
                            #self.resizeThreshImages(image)                            
                            #self.listOfRawImages.append(image)  
                            i+=1
                        
                        cap.release()
                        cv2.destroyAllWindows()


                    if 'Mask' in path:
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            cv2.imwrite(os.path.join(pathToFolderMask, "maskframe%d.jpg" % i), image)
                            #image = self.resizeThreshImages(image)
                            #self.operation1_listFull.append(image)
                            #cv2.imshow('binary_mask', image)
                            #cv2.waitKey(20)
                            i+=1
                        
                        cap.release()
                        cv2.destroyAllWindows()  
                        

            if 'Dataset3' in root:
                for file in files:
                    path = os.path.join(root, file)
                    pathToFolderRaw = './Data/Segmentation_Robotic_Training/Training/Dataset3/Raw'
                    pathToFolderMask = './Data/Segmentation_Robotic_Training/Training/Dataset3/Mask'

                    if 'Video' in path:
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            cv2.imwrite(os.path.join(pathToFolderRaw, "rawframe%d.jpg" % i), image)
                            #self.resizeThreshImages(image)
                            #self.listOfRawImages.append(image)  
                            i+=1
                        
                        cap.release()
                        cv2.destroyAllWindows()


                    if 'Mask' in path:
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            cv2.imwrite(os.path.join(pathToFolderMask, "maskframe%d.jpg" % i), image)
                            #image = self.resizeThreshImages(image)
                            #self.operation1_listFull.append(image)
                            i+=1
                        
                        cap.release()
                        cv2.destroyAllWindows() 

            if 'Dataset4' in root:
                for file in files:
                    path = os.path.join(root, file)
                    pathToFolderRaw = './Data/Segmentation_Robotic_Training/Training/Dataset4/Raw'
                    pathToFolderMask = './Data/Segmentation_Robotic_Training/Training/Dataset4/Mask'

                    if 'Video' in path:
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            cv2.imwrite(os.path.join(pathToFolderRaw, "rawframe%d.jpg" % i), image)
                            #self.resizeThreshImages(image)
                            #self.listOfRawImages.append(image)  
                            i+=1
                        
                        cap.release()
                        cv2.destroyAllWindows()


                    if 'Mask' in path:
                        
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            cv2.imwrite(os.path.join(pathToFolderMask, "maskframe%d.jpg" % i), image)
                            
                            #image = self.resizeThreshImages(image)
                            #self.operation1_listFull.append(image)
                            i+=1
                        

                        cap.release()
                        cv2.destroyAllWindows()         
                        #np.save('RoboticDataSet',np.array(self.operation1_listFull))
                        #np.save('RoboticDataSet_Raw',np.array(self.listOfRawImages))
                        #print(np.array(self.listOfRawImages).shape)
                          

                 


load = DataLoader()

masks_class, masks_inst, raws = load.loadDataRigid('./Data')

load.loadDataRobotics('./Data/Segmentation_Robotic_Training')

print(masks_class.shape, masks_inst.shape, raws.shape)
