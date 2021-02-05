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
        self.masks = [] 
    
    def resizeImages(self,inputImage, width, height):
        resize(inputImage,(width,height))
    
    def normalize(self,inputImage): 
        normalize(inputImage) 
    
    # return numpy array of images (raws and masks) 
    # raw shape : (160, 480, 640, 3)
    # masks shape : (320, 480, 640, 3)
    def loadData(self,path):
        raw_ImageList = []
        mask_ImageList = []
            
        for (root, dirs, files) in walk(path):

            # sort file names by number, alphapets,...
            dirs.sort()
            files.sort()
            
            if 'Masks' in root:
                for file in files:
                    path = os.path.join(root, file)
                    image = cv2.imread(path) 
                    mask_ImageList.append(image)
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
        self.masks = np.array(mask_ImageList)

        return self.masks, self.raws
        
        
        
    
   

        


load = DataLoader()

rawImages, masks = load.loadData('./Data')
