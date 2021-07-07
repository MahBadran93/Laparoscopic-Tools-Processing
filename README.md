# Advance processing of laparoscopy tools 
## GitHub Repositories used : 
  - https://github.com/matterport/Mask_RCNN 
    - To train the object detector (Mask R-CNN), code from this reposotery was used in our implementation. 
  - https://github.com/kamruleee51/ART-Net 
    - This reposotery is used to implement ART-Net. To generate semantic segmentation and geometric primitivs probability segmentation maps.
    
## Requernments:
  - Tensorflow 2.4.1 
  - Ubuntu 18 

## System 
  - Core i9
  - Geforce RTX 3080 
  - 32 gb RAM
  
## Surgical tool detection 
  - To detect surgical tools, we used Mask-RCNN. The base code used is from the repository referenced above.  
  - To run the code: 
    - The dataset is annotated using the [supervisely web platform](https://supervise.ly/)
    - Load the dataset. [surgAR_Load_Tool_All.py](https://github.com/MahBadran93/laparoscopic-Tools-Segmentation/blob/main/Mask_RCNN/surgAR_Load_Tool_All.py) file will be used to load the data to be suiable input to Mask-RCNN.    
    - Run [Main_Tool](https://github.com/MahBadran93/laparoscopic-Tools-Segmentation/blob/main/Mask_RCNN/Main_Tool.ipynb)    

## Geometric Primitives extraction 

  - Run **Main.py** file 
  - The original images and the result can be found in the **data** folder. 
  - If you want to test a new image, you can go to the the end of the code in the **Main.py** file and change the **path1** variable to the new image path.
  
## Results: 

 <p align="center">
    <img  src = "data/originalImages/original.jpg" width=300> <br>
     <em>Original Image</em>
 </p>
  <p align="center">
    <img  src = "data/homomorpicFilterImages/image_homomorphic2.jpg" width=300> <br>
     <em>Filtered Image</em>
 </p>
