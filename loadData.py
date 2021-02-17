import numpy as np
import  cv2
dataset_raw = np.load('./RoboticDataSet_Raw.npy')


print(dataset_raw.shape)

for i in range(len(dataset_raw)):
    cv2.imshow('masks',dataset_raw[i])
    cv2.waitKey(20)
