
import numpy as np
import  glob
import pandas as pd
import os
# path of text files 
pathtext = '/home/mahmoud/Desktop/AnnotateFullArtnet/coordinates/'
texts =glob.glob('/home/mahmoud/Desktop/AnnotateFullArtnet/coordinates/*.txt')
pathcsv = '/home/mahmoud/Desktop/AnnotateFullArtnet/coordinatecsv/'
def ToCSV(path, filename):    
    read_file = pd.read_csv(path , header=None, delimiter=',')
    read_file.columns = ['X','Y']
    read_file.to_csv(pathcsv + filename + '.csv', index=None)
    return read_file


for path in texts:
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    ToCSV(path, filename)
