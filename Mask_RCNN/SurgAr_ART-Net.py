import os 
import glob
import json
import numpy as np
import cv2 
from skimage.draw import polygon, line, circle, line_aa
import matplotlib.pyplot as plt

# path to dataset 
dataset_dir = '/home/mahmoud/Desktop/newdataall/'

def load_SurgAR(dataset_dir):
        """Load a subset of the Tool dataset.
        dataset_dir: Root directory of the dataset. '/home/mahmoud/Desktop/newdataall/'
        subset: Subset to load: train or val
        """
        linesFolder = ''
        tipFolder = '' 
        maskFolder = ''

        dataset_dir = os.path.join(dataset_dir, 'train' + '/ann/')

        # iterate over all the json files in train/img folder because we are using supervisely dataset
        # and they are separating annotations(json) for each image 
        annot=glob.glob(dataset_dir + '/*.json')
        annot.sort()
        
        # It includes each image path with its all instances(classes) points and names 
        

        # inintialize image path     
        imagePath = '' 
        # iterate each json file(each image)

                        
        # Iterate over all json files for all images in the dataset                 
        for path in annot:
            # read json file
            annotations = json.load(open(path))

            lenMid = 0
            lenEdge = 0 
            lenTip = 0
            lenClasses = 0 
            
            for cl in annotations["objects"]:
                if cl['classTitle'] == 'mid-line':
                    lenMid = lenMid + 1
                if cl['classTitle'] == 'edge-line':
                    lenEdge = lenEdge + 1    
                if cl['classTitle'] == 'tool-tip':
                    lenTip = lenTip + 1   
                lenClasses = lenClasses + 1

            lenTool = lenClasses - (lenMid + lenEdge + lenTip)   

            print(lenMid,lenEdge,lenTip,lenTool )    
     
            # read height and width for each image
            width=annotations['size']['width']
            height=annotations['size']['height']

            # Create zero mask with (width, heigh, number of object of the image 
            mask = np.zeros((height, width, lenClasses), dtype=np.uint8) 
            mask_lines = np.zeros((height, width),
                        dtype=np.uint8)
            mask_tip = np.zeros((height, width),
                        dtype=np.uint8)            

            # Iterate over all objects (tools and geometric primitves for each image)    
            for obj in annotations['objects']:
                i = 0
                x = []
                y = []
                for points in obj['points']['exterior']:
                    x.append(points[0])
                    y.append(points[1])
                if obj['classTitle'] == 'tool-tip':   # len(obj['points']['exterior']) == 1 and 
                    #rr,cc = circle(y[0],x[0],5)
                    mask_tip = cv2.circle(mask_tip, (y[0],x[0]), radius=0, color=255, thickness=100)
                    mask_tip = cv2.distanceTransform(mask_tip,cv2.DIST_L2,5)
                    mask_tip = cv2.normalize(mask_tip, mask_tip, 0, 1.0, cv2.NORM_MINMAX)
                    mask[:,:,i] = np.uint8(mask_tip*255)
                    #mask[rr,cc,i] = 1
                if obj['classTitle'] == 'mid-line':   
                    mask_lines = cv2.line(mask_lines,(x[0],y[0]),(x[1],y[1]),1,15)
                    mask_lines = cv2.distanceTransform(mask_lines,cv2.DIST_L2,5)
                    mask_lines = cv2.normalize(mask_lines, mask_lines, 0, 1.0, cv2.NORM_MINMAX)
                    mask[:,:,i] = np.uint8(mask_lines*255)
                    mask_lines = np.zeros((height,width),
                            dtype=np.uint8)    
                if obj['classTitle'] == 'edge-line':   
                    mask_lines = cv2.line(mask_lines,(x[0],y[0]),(x[1],y[1]),1,15)
                    mask_lines = cv2.distanceTransform(mask_lines,cv2.DIST_L2,5)
                    mask_lines = cv2.normalize(mask_lines, mask_lines, 0, 1.0, cv2.NORM_MINMAX)
                    mask[:,:,i] = np.uint8(mask_lines*255)
                    mask_lines = np.zeros((height,width),
                            dtype=np.uint8)  
                else:
                    rr, cc = polygon(y,x)
                    rr[rr > mask.shape[0]-1] = mask.shape[0]-1
                    cc[cc > mask.shape[1]-1] = mask.shape[1]-1  
                    mask[rr, cc, i] = 1            

                plt.imshow(mask[:,:,i])
                plt.show()
                i = i + 1 
            break
        
            
load_SurgAR(dataset_dir)

def save_masks(self, image_id):

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