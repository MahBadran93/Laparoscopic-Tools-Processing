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
    
    def resizeImages(self,inputImage, width, height):
        resize(inputImage,(width,height))
    
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
    
    
    
    def loadDataRobotics(self,path):
        
        def readFrames(path):
            images = []
            for filename in os.listdir(path):
                images.append(os.path.join(path,filename))
            return images
        
        
        
    
   

        


load = DataLoader()

masks_class, masks_inst, raws = load.loadDataRigid('./Data')

print(masks_class.shape, masks_inst.shape, raws.shape)
