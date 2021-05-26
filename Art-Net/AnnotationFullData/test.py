import cv2
path = '/home/mahmoud/Desktop/AnnotateFullArtnet/Data/Edgeline1/image43.png'
path2 = '/home/mahmoud/Desktop/laparoscopic-Tools-Segmentation/DataFinal/Train/Train_Positive_EdgeLine_1/Train_Pos_sample_0002_EdgeLine_1.png'
image0 = cv2.imread(path2)
image = cv2.imread(path2,-1)
image = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
print(image0.shape)
print(image.shape)
'''
img_pathmidP=image/image.max()
img_pathmidP = img_pathmidP[:,:,0:1]
print(img_pathmidP.shape)    
'''

