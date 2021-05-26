import imageio
import numpy as np
import imgaug as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import cv2

# Load an example image (uint8, 128x128x3).
path = '/home/badran/Desktop/AnnotationFullData/TestAnnotation/PositiveImage/image43.png'
image = cv2.imread(path)

# Create an example mask (bool, 128x128).
# Here, we arbitrarily place a square on the image.
pathmask = '/home/badran/Desktop/AnnotationFullData/TestAnnotation/Edgeline1/image43.png'
mask = cv2.imread(pathmask)

segmap = SegmentationMapsOnImage(mask, shape=image.shape)



# Define our augmentation pipeline.
seq = iaa.Sequential([
    iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
    iaa.Sharpen((0.0, 1.0)),       # sharpen the image
    iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
    iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
    
], random_order=True)

# Augment images and segmaps.
images_aug = []
segmaps_aug = []
for _ in range(5):
    images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
    images_aug.append(images_aug_i)
    segmaps_aug.append(segmaps_aug_i)


    