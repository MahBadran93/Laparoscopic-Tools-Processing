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
        self.mergeLeftRightList = []

        # Training Mask images list 
        self.maskTraining = []
        # Training Raw images list, rawTraining
        self.rawTraining = []
        # Testing Masks 
        self.gtTesting = [] 
        # raw images testing 
        self.rawTesting = []

        # Training data raw for proposed dataset 
        self.proposedDataSetRaw = []
        # Training data masks for the proposed dataset 
        self.proposedDataSetMask = []


    #convert from uint8 to float and resize 
    def resizeThreshImages(self,inputImage, width = 192, height = 256):
        #gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
        image = resize(inputImage,(width,height))
        thresh = 0
        im_bw = cv2.threshold(image, thresh, 1, cv2.THRESH_BINARY)[1]
        image = np.array(im_bw)
        return image
 
    # return numpy array of rigid segmentation data provided by EncoVis (raws and instrument masks and class masks ) 
    # raw shape : (160, 480, 640, 3)
    # class masks shape : (160, 480, 640, 3)
    # instrument masks shape : (160 , 480, 640, 3)

    def loadRoboticTestData(self, path): 
      #pathToRaw = '/content/drive/MyDrive/MasterThesis /Data/RoboticInstruments_Test/Dataset2/Raw_Test'
      #pathToGT = '/content/drive/MyDrive/MasterThesis /Data/RoboticInstruments_Test/GT/Dataset2/GT_Masks'
      arrayLeftRight1 = [] 
      arrayLeftRight5 = []
      for (root, dirs, files) in walk(path):
          if 'GT' in root:              
              if 'Dataset1' in root:
                  for file in files:
                      path = os.path.join(root, file)
                      cap= cv2.VideoCapture(path)
                      i=0
                      while(cap.isOpened()):
                          ret, image = cap.read()
                          if ret == False:
                              break
                          image = self.resizeThreshImages(image)
                          arrayLeftRight1.append(image)                            
                          i+=1
                      cap.release()
                      cv2.destroyAllWindows()   
                      
                  #merge left nd right tools in one mask
                  #print(np.array(arrayLeftRight).shape) 
                  num = int(len(arrayLeftRight1)/2)         
                  for i in range(num):
                      self.gtTesting.append(arrayLeftRight1[i] + arrayLeftRight1[i + num]) 
              if 'Dataset2' in root:
                  for file in files:
                      path = os.path.join(root, file)
                      cap= cv2.VideoCapture(path)
                      i=0
                      while(cap.isOpened()):
                          ret, image = cap.read()
                          if ret == False:
                              break
                          image = self.resizeThreshImages(image)
                          self.gtTesting.append(image)                            
                          i+=1
                      cap.release()
                      cv2.destroyAllWindows()   
              if 'Dataset3' in root:
                  for file in files:
                      path = os.path.join(root, file)
                      cap= cv2.VideoCapture(path)
                      i=0
                      while(cap.isOpened()):
                          ret, image = cap.read()
                          if ret == False:
                              break
                          image = self.resizeThreshImages(image)
                          self.gtTesting.append(image)                            
                          i+=1
                      cap.release()
                      cv2.destroyAllWindows()            
              if 'Dataset4' in root:
                  for file in files:
                      path = os.path.join(root, file)
                      cap= cv2.VideoCapture(path)
                      i=0
                      while(cap.isOpened()):
                          ret, image = cap.read()
                          if ret == False:
                              break
                          image = self.resizeThreshImages(image)
                          self.gtTesting.append(image)                            
                          i+=1
                      cap.release()
                      cv2.destroyAllWindows()   
              if 'Dataset5' in root:
                  for file in files:
                      path = os.path.join(root, file)
                      cap= cv2.VideoCapture(path)
                      i=0
                      while(cap.isOpened()):
                          ret, image = cap.read()
                          if ret == False:
                              break
                          image = self.resizeThreshImages(image)
                          arrayLeftRight5.append(image)                            
                          i+=1
                      cap.release()
                      cv2.destroyAllWindows()   
                      
                  #merge left nd right tools in one mask
                  #print(np.array(arrayLeftRight).shape) 
                  num = int(len(arrayLeftRight5)/2)         
                  for i in range(num):
                      self.gtTesting.append(arrayLeftRight5[i] + arrayLeftRight5[i + num]) 
              if 'Dataset6' in root:
                  for file in files:
                      path = os.path.join(root, file)
                      cap= cv2.VideoCapture(path)
                      i=0
                      while(cap.isOpened()):
                          ret, image = cap.read()
                          if ret == False:
                              break
                          image = self.resizeThreshImages(image)
                          self.gtTesting.append(image)                            
                          i+=1
                      cap.release()
                      cv2.destroyAllWindows()   
               
                    
          else:
              pass
              '''
              if 'Dataset1' in root:
                  for file in files:
                      if file == 'Video.avi':
                          path = os.path.join(root, file)
                          print(path)
                          cap= cv2.VideoCapture(path)
                          i=0
                          while(cap.isOpened()):
                              ret, image = cap.read()
                              if ret == False:
                                  break
                              image = resize(image,(192,256)) 
                              image = np.array(image) 
                              #cv2.imwrite(os.path.join(pathToRaw, "rawtest2%d.jpg" % i), image)  
                              self.rawTesting.append(image)
                              i+=1
                          print(i)
                          cap.release()
                          cv2.destroyAllWindows()
              if 'Dataset2' in root:
                  for file in files:
                      if file == 'Video.avi':
                          path = os.path.join(root, file)
                          print(path)
                          cap= cv2.VideoCapture(path)
                          i=0
                          while(cap.isOpened()):
                              ret, image = cap.read()
                              if ret == False:
                                  break
                              image = resize(image,(192,256)) 
                              image = np.array(image) 
                              #cv2.imwrite(os.path.join(pathToRaw, "rawtest2%d.jpg" % i), image)  
                              self.rawTesting.append(image)
                              i+=1
                          print(i)
                          cap.release()
                          cv2.destroyAllWindows()
              if 'Dataset3' in root:
                  for file in files:
                      if file == 'Video.avi':
                          path = os.path.join(root, file)
                          print(path)
                          cap= cv2.VideoCapture(path)
                          i=0
                          while(cap.isOpened()):
                              ret, image = cap.read()
                              if ret == False:
                                  break
                              image = resize(image,(192,256)) 
                              image = np.array(image) 
                              #cv2.imwrite(os.path.join(pathToRaw, "rawtest2%d.jpg" % i), image)  
                              self.rawTesting.append(image)
                              i+=1
                          print(i)
                          cap.release()
                          cv2.destroyAllWindows()  
              if 'Dataset4' in root:
                  for file in files:
                      if file == 'Video.avi':
                          path = os.path.join(root, file)
                          print(path)
                          cap= cv2.VideoCapture(path)
                          i=0
                          while(cap.isOpened()):
                              ret, image = cap.read()
                              if ret == False:
                                  break
                              image = resize(image,(192,256)) 
                              image = np.array(image) 
                              #cv2.imwrite(os.path.join(pathToRaw, "rawtest2%d.jpg" % i), image)  
                              self.rawTesting.append(image)
                              i+=1
                          print(i)
                          cap.release()
                          cv2.destroyAllWindows()                          
              if 'Dataset5' in root:
                  for file in files:
                      if file == 'Video.avi':
                          path = os.path.join(root, file)
                          print(path)
                          cap= cv2.VideoCapture(path)
                          i=0
                          while(cap.isOpened()):
                              ret, image = cap.read()
                              if ret == False:
                                  break
                              image = resize(image,(192,256)) 
                              image = np.array(image) 
                              #cv2.imwrite(os.path.join(pathToRaw, "rawtest2%d.jpg" % i), image)  
                              self.rawTesting.append(image)
                              i+=1
                          print(i)
                          cap.release()
                          cv2.destroyAllWindows()
              if 'Dataset6' in root:
                  for file in files:
                      if file == 'Video.avi':
                          path = os.path.join(root, file)
                          print(path)
                          cap= cv2.VideoCapture(path)
                          i=0
                          while(cap.isOpened()):
                              ret, image = cap.read()
                              if ret == False:
                                  break
                              image = resize(image,(192,256)) 
                              image = np.array(image) 
                              #cv2.imwrite(os.path.join(pathToRaw, "rawtest2%d.jpg" % i), image)  
                              self.rawTesting.append(image)
                              i+=1
                          cap.release()
                          cv2.destroyAllWindows()
              np.savez_compressed('./Data/rawTesting.npy', self.rawTesting)  
              '''  
        
        #........................ Masks Robotics..................................

      print(np.array(self.gtTesting).shape)        
      np.savez_compressed('./Data/gtTesting.npy', self.gtTesting)   


    def loadRoboticTrainData(self,path):
        count = 0
        for (root, dirs, files) in walk(path):  
            if 'Dataset1' in root:
                count +=1
                for file in files:
                    path = os.path.join(root, file)
                    #pathToFolderRaw = './Data/Segmentation_Robotic_Training/Training/Dataset1/Raw'
                    #pathToFolderLeft = './Data/Segmentation_Robotic_Training/Training/Dataset1/Left'
                    #pathToFolderRight = './Data/Segmentation_Robotic_Training/Training/Dataset1/Right'
                    if 'Video' in path:
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            #cv2.imwrite(os.path.join(pathToFolderRaw, "rawframe%d.jpg" % i), image)
                            image = resize(image,(192,256))
                            image - np.array(image)
                            self.rawTraining.append(image)                            
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
                            #cv2.imwrite(os.path.join(pathToFolderLeft, "leftmaskframe%d.jpg" % i), image)
                            image = self.resizeThreshImages(image)
                            self.mergeLeftRightList.append(image)
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
                            #cv2.imwrite(os.path.join(pathToFolderRight, "rightmaskframe%d.jpg" % i), image)
                            image = self.resizeThreshImages(image)
                            self.mergeLeftRightList.append(image)
                            i+=1
                        cap.release()
                        cv2.destroyAllWindows()

                        #merge left nd right tools in one mask
                        num = int(len(self.mergeLeftRightList)/2)
                        for i in range(num):
                            self.maskTraining.append(self.mergeLeftRightList[i] + self.mergeLeftRightList[i + num])

            if 'Dataset2' in root:
                count +=1
                for file in files:
                    path = os.path.join(root, file)
                    #pathToFolderRaw = './Data/Segmentation_Robotic_Training/Training/Dataset2/Raw'
                    #pathToFolderMask = './Data/Segmentation_Robotic_Training/Training/Dataset2/Mask'
                    if 'Video' in path:
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            #cv2.imwrite(os.path.join(pathToFolderRaw, "rawframe%d.jpg" % i), image)
                            image = resize(image,(192,256))
                            image - np.array(image)  
                            self.rawTraining.append(image)
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
                            #cv2.imwrite(os.path.join(pathToFolderMask, "maskframe%d.jpg" % i), image)
                            image = self.resizeThreshImages(image)
                            self.maskTraining.append(image)
                            i+=1
                        
                        cap.release()
                        cv2.destroyAllWindows()  
                        
            if 'Dataset3' in root:
                count +=1
                for file in files:
                    path = os.path.join(root, file)
                    #pathToFolderRaw = './Data/Segmentation_Robotic_Training/Training/Dataset3/Raw'
                    #pathToFolderMask = './Data/Segmentation_Robotic_Training/Training/Dataset3/Mask'

                    if 'Video' in path:
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            #cv2.imwrite(os.path.join(pathToFolderRaw, "rawframe%d.jpg" % i), image)
                            image = resize(image,(192,256))
                            image - np.array(image)  
                            self.rawTraining.append(image)
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
                            #cv2.imwrite(os.path.join(pathToFolderMask, "maskframe%d.jpg" % i), image)
                            image = self.resizeThreshImages(image)
                            self.maskTraining.append(image)
                            i+=1 
                        cap.release()
                        cv2.destroyAllWindows() 

            if 'Dataset4' in root:
                count +=1
                for file in files:
                    path = os.path.join(root, file)
                    #pathToFolderRaw = './Data/Segmentation_Robotic_Training/Training/Dataset4/Raw'
                    #pathToFolderMask = './Data/Segmentation_Robotic_Training/Training/Dataset4/Mask'

                    if 'Video' in path:
                        cap= cv2.VideoCapture(path)
                        i=0
                        while(cap.isOpened()):
                            ret, image = cap.read()
                            if ret == False:
                                break
                            #cv2.imwrite(os.path.join(pathToFolderRaw, "rawframe%d.jpg" % i), image)
                            image = resize(image,(192,256))
                            image - np.array(image)
                            self.rawTraining.append(image)  
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
                            #cv2.imwrite(os.path.join(pathToFolderMask, "maskframe%d.jpg" % i), image)  
                            image = self.resizeThreshImages(image)
                            self.maskTraining.append(image)
                            i+=1
                        

                        cap.release()
                        cv2.destroyAllWindows()         
                        np.savez_compressed('./Data/maskTrining.npy',np.array(self.maskTraining))
                        np.savez_compressed('./Data/rawTrining.npy',np.array(self.rawTraining))
                        print(np.array(self.rawTraining).shape,np.array(self.maskTraining) )
                          

                 


load = DataLoader()

# make training data set 
load.loadRoboticTrainData('./Data/Segmentation_Robotic_Training/Training/')

# mke testing dataset 
load.loadRoboticTestData('./Data/Robotic Instruments_Testing/')