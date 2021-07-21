import os 
import glob 
import json
import cv2 
import os, sys 
import shutil 

# Dataset paths 
# Dataset paths 
dataset_ann_dir = '/home/mahmoud/Downloads/InstrumentsAll/InstrumentsAll/dataset/ann/'
dataset_img_dir = '/home/mahmoud/Downloads/InstrumentsAll/InstrumentsAll/dataset/img/'
img_trainPath = '/home/mahmoud/Desktop/Mask_RCNN/samples/Tool/dataset/All_Data/New/simple/train/img'
ann_trainPath = '/home/mahmoud/Desktop/Mask_RCNN/samples/Tool/dataset/All_Data/New/simple/train/ann'
img_valPath = '/home/mahmoud/Desktop/Mask_RCNN/samples/Tool/dataset/All_Data/New/simple/val/img'
ann_valPath = '/home/mahmoud/Desktop/Mask_RCNN/samples/Tool/dataset/All_Data/New/simple/val/ann'
ann_testPath = '/home/mahmoud/Desktop/Mask_RCNN/samples/Tool/dataset/All_Data/New/simple/test/ann'
img_testPath = '/home/mahmoud/Desktop/Mask_RCNN/samples/Tool/dataset/All_Data/New/simple/test/img'
# split dataset from supervisel to train and validation (suitable input for Mask R-CNN)
def splitTrainVal(dataset_ann_dir):
    # iterate over all the json files in train/img folder because we using supervisely dataset
    # and they are separating annotations(json) for each image 
    annot=glob.glob(dataset_ann_dir + '/*.json')
    annot.sort()
    # It includes each image path with its all instances(classes) points and names 
    # iterate each json file(each image)
    count = 0
    for path in annot:
        #print(path)
        annotations = json.load(open(path))
        #temp = splitall(path)
        filename = os.path.basename(path)
        filename = os.path.splitext(filename)[0]
        print(path)
        imgPath = dataset_img_dir + filename
        # 10: when augmentation, 3 when no augmentation
        for status in annotations['tags']:
            if status['name'] == 'val' and count%3 == 0:
                shutil.move(path, ann_testPath)
                shutil.move(imgPath, img_testPath)
            else:        
                if status['name'] == 'train':
                    shutil.move(path, ann_trainPath)
                    shutil.move(imgPath, img_trainPath)
                elif status['name'] == 'val':
                    shutil.move(path, ann_valPath)
                    shutil.move(imgPath, img_valPath)
            count = count + 1     

 



splitTrainVal(dataset_ann_dir)