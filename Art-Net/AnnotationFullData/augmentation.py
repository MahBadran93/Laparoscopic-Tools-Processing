import imageio
from imgaug.augmenters.geometric import PerspectiveTransform
from matplotlib import scale
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import cv2
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import resize
import albumentations as A
import cv2
import glob
import os



def Augment(image, masks,  pathO, pathE1, pathE2, pathMid, pathMask, pathT, filename, flag, num_aug = 5):
    # image : original image converted to grayscale. image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # masks : all the masks corresponding to the original image. masks = [mask1, mask2, ...]  
    # num_image : number of annotations per image  

    transform = A.Compose([
        #A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.augmentations.geometric.resize.RandomScale(scale_limit([0.8,1.2])),
        #A.PerspectiveTransform([0.01,0.1]),
        A.RandomBrightnessContrast(p=0.2),
    ])

    augmentedOrigImages = []
    augmenedMasks = []
    for i in range(5):
        transformed = transform(image=image, masks=masks)
        transformed_image = transformed['image']
        transformed_masks = transformed['masks']
        augmentedOrigImages.append(transformed_image)
        augmenedMasks.append(transformed_masks)
        # save augmented images 
        writePath = pathO + '/'  + 'image{}'.format(flag) + '_aug{}.png'.format(i)
        #cv2.imwrite(writePath, transformed_image)   

        # save correspnding masks 
        writePathE1 = pathE1 + '/' + 'image{}'.format(flag) +'_aug{}.png'.format(i)
        #cv2.imwrite(writePathE1, transformed_masks[0])

        writePathE2 = pathE2 + '/' + 'image{}'.format(flag) +'_aug{}.png'.format(i)
        #cv2.imwrite(writePathE2, transformed_masks[1])

        writePathMask = pathMask + '/' + 'image{}'.format(flag) +'_aug{}.png'.format(i)
        #cv2.imwrite(writePathMask, transformed_masks[3])

        writePathMid = pathMid + '/' + 'image{}'.format(flag) +'_aug{}.png'.format(i)
        #cv2.imwrite(writePathMid, transformed_masks[2])

        writePathTip = pathT + '/' + 'image{}'.format(flag) +'_aug{}.png'.format(i)
        cv2.imwrite(writePathTip, transformed_masks[4])



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

XX=zip(TI, TM, TU, TL, TT, TMas)
i=0
flag = 0
#kernel2=disk(4)

for pathorigP, pathmidP, pathupP, pathlowP, pathtipP, pathmasP in XX:
    # pathO
    pathO = os.path.dirname(pathorigP)
    pathO = os.path.splitext(pathO)[0]
    filename = os.path.basename(pathorigP)
    filename = os.path.splitext(filename)[0]
    # pathE1
    pathE1 = os.path.dirname(pathupP)
    pathE1 = os.path.splitext(pathE1)[0]
    #print(pathE1)
    # pathE2
    pathE2 = os.path.dirname(pathlowP)
    pathE2 = os.path.splitext(pathE2)[0]
    #print(pathE2)
    # pathMid
    pathMid = os.path.dirname(pathmidP)
    pathMid = os.path.splitext(pathMid)[0]
    #print(pathMid)
    #pathMask
    pathMask = os.path.dirname(pathmasP)
    pathMask = os.path.splitext(pathMask)[0]
    #print(pathMask)
    #pathT
    pathT = os.path.dirname(pathtipP)
    pathT = os.path.splitext(pathT)[0]
    #print(pathT)

    # inpu image to augment 
    image = cv2.imread(pathorigP)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # corresponding masks to augment 
    maskE1 = cv2.imread(pathupP)
    maskE2 = cv2.imread(pathlowP)
    maskMid = cv2.imread(pathmidP)
    maskMask = cv2.imread(pathmasP)
    maskT = cv2.imread(pathtipP)
    masks = [maskE1, maskE2, maskMid, maskMask, maskT]
    
    Augment(image,masks, pathO, pathE1, pathE2, pathMid, pathMask, pathT, filename,flag)
    flag = flag + 1
