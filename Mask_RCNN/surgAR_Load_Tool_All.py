"""
Mask R-CNN
Train on the surgical tools(Encov) dataset
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 Tool.py train --dataset=/home/datascience/Workspace/maskRcnn/Mask_RCNN-master/samples/bottle/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 Tool.py train --dataset=/path/to/bottle/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 Tool.py train --dataset=/path/to/bottle/dataset --weights=imagenet

    # Apply color splash to an image
    python3 Tool.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 Tool.py splash --weights=last --video=<URL or path to file>
"""
import tensorflow as tf
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
import pandas as pd 
import glob
from skimage.draw import polygon, line, circle, line_aa
from imgaug import augmenters as iaa
from sklearn.utils import class_weight
from collections import Counter


# Root directory of the project
ROOT_DIR = os.path.abspath("/home/mahmoud/Desktop/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 30  # Background + other classes
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # validation steps per epoch 
    VALIDATION_STEPS = 100
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    LEARNING_RATE =  0.001
    BACKBONE = "resnet101" # resnet101, resnet50
    #WEIGHT_DECAY = 0.01 # 0.005, 0.001
    LEARNING_MOMENTUM = 0.75


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the bottle dataset.
        dataset_dir: Root directory of the dataset. '/home/mahmoud/Desktop/Mask_RCNN-master/dataset/CHUDataset/'
        subset: Subset to load: train or val
        """

        # add classes titles and tags.
        info = json.load(open(os.path.join(dataset_dir, "meta.json")))
        i = 0
        for cl in info['classes']:
            i = i + 1 
            # Add classes. we can replace cl with the cl['id'] from the meta json 
            self.add_class("Tool", i, cl['title'])
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset + '/ann/')

        # iterate over all the json files in train/img folder because we using supervisely dataset
        # and they are separating annotations(json) for each image 
        annot=glob.glob(dataset_dir + '/*.json')
        annot.sort()
        
        # It includes each image path with its all instances(classes) points and names 
        

        # inintialize image path     
        imagePath = '' 
        # iterate each json file(each image)
        for path in annot:
            polygons ={
            "classTitle": [],
            "classID": [],
            "polygonMaskPoints": []
            }
            # Get image path because the path is in spearate folder and is not written in the json file 
            imagePath = path.split("/")
            for i in range (len(imagePath)):
                if imagePath[i] == 'ann':
                    imagePath[i] = 'img'
            imagePath[len(imagePath)-1] = imagePath[len(imagePath)-1].replace('.json', '')
            imagePath = '/'.join(imagePath)

            # clear the names and instances for each iteration to prevent stacking all the data 
            #polygons["classTitle"].clear()
            #polygons["polygonMaskPoints"].clear() 
            # Read json file 
            annotations = json.load(open(path))
            for obj in annotations['objects']:
                polygons['classTitle'].append(obj['classTitle'])
                polygons['classID'].append(obj['classId'])
                polygons['polygonMaskPoints'].append(obj['points'])
            # save image with all needed details  
              
            self.add_image(
                    "Tool", 
                    image_id=path,  # use path as a unique image id
                    path=imagePath,
                    width=annotations['size']['width'],
                    height=annotations['size']['height'],
                    polygons=polygons)  
              

    def load_mask(self, image_id):

        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Tool dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Tool":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        classes = []
        info = self.image_info[image_id]
        classID = []
        # Add classes IDs to list                 
        for cl in range(len(info["polygons"]["classTitle"])):
          classes.append(info["polygons"]["classTitle"][cl])
          #classID.append(info["polygons"]["classID"][cl])

        mask = np.zeros([info["height"], info["width"], len(classes)],
                        dtype=np.uint8)
        mask_lines = np.zeros((info["height"], info["width"]),
                        dtype=np.uint8)     
        # Geometry Config 
        toolTipThickness = 10

        # add class ids 
        classinfo = info['class_info']
        for name in classes:
          classID.append(list(filter(lambda classinfo: classinfo['name'] == name, classinfo))[0]['id'])
        
        # Convert classes ides list to numpy array 
        classesIdes = np.array(classID, dtype=np.int32)
        
        i = 0 
        for instance in info["polygons"]["polygonMaskPoints"]:
            x = []
            y = []
            for points in instance['exterior']:
                #joined_string = list(points[0])
                x.append(points[0])
                y.append(points[1])
            if(len(instance['exterior']) == 2 ):
                '''
                rr, cc, val = line_aa(y[0],x[0], y[1],x[1])
                rr[rr > mask.shape[0]-1] = mask.shape[0]-1
                cc[cc > mask.shape[1]-1] = mask.shape[1]-1
                mask[rr, cc, i] = 1
                '''
                mask[:,:,i] = cv2.line(mask_lines,(x[0],y[0]),(x[1],y[1]),1,5)
                mask_lines = np.zeros((info["height"], info["width"]),
                        dtype=np.uint8)
	          # tool tip 
            elif(len(instance['exterior'])==1):
                rr,cc = circle(y[0],x[0],toolTipThickness)
                mask[rr,cc,i] = 1
            else:      
                rr, cc = polygon(y,x)
                rr[rr > mask.shape[0]-1] = mask.shape[0]-1
                cc[cc > mask.shape[1]-1] = mask.shape[1]-1  
                mask[rr, cc, i] = 1
            i = i + 1        

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), classesIdes

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Tool":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""

    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # add class weights 
    CLASS_WEIGHTS = { 0:189, 1:22, 2:1, 3:40, 4:28, 5:85, 6:40, 7:63, 8:42, 9:5 }

    model_inference = modellib.MaskRCNN(mode="inference", config=CustomConfig(),
                                model_dir=args.logs)
     # Custom callback to calculate mAP for each epich during training 
    mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model,
                            model_inference, dataset_val, calculate_map_at_every_X_epoch=10, log=args.logs, verbose=1)

    # add online augmentation 
    augmentation = iaa.SomeOf((0, 3), [
      iaa.Fliplr(0.5),
      iaa.Flipud(0.5),
      iaa.OneOf([iaa.Affine(rotate=90),
                 iaa.Affine(rotate=180),
                 iaa.Affine(rotate=270)]),
      iaa.Multiply((0.8, 1.5)),
      iaa.GaussianBlur(sigma=(0.0, 5.0))
  ])


    def compute_weights(CLASS_WEIGHTS):
        mean = np.array(list(CLASS_WEIGHTS.values())).mean() # sum_class_occurence / nb_classes
        max_weight = np.array(list(CLASS_WEIGHTS.values())).max()
        CLASS_WEIGHTS.update((x, float(max_weight/(y))) for x, y in CLASS_WEIGHTS.items())
        CLASS_WEIGHTS=dict(sorted(CLASS_WEIGHTS.items()))
        return CLASS_WEIGHTS
    
    class_weights = compute_weights(CLASS_WEIGHTS)

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                layers='heads',
                #augmentation=augmentation,
                #class_weight=class_weights,
                #custom_callbacks=[mean_average_precision_callback]
               )

    
    '''
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='all')
    '''
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

