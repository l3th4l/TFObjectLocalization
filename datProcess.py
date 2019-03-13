import xml.etree.ElementTree as ET 
from skimage import io 
from skimage import util as ut 
from matplotlib import pyplot as plt 
from PIL import Image
import numpy as np 
import pandas as pd
import os

#crop window size 
cSize = 32 #64 pixels 

dat_path = './Dataset/training_data'

try: 
        os.mkdir(dat_path + '/processed')
except:
        pass

def process_dat(filename, path):
        xml_file = '%s/labels/%s.xml' % (path, filename[ : -4])
        print(filename[ : -4])
        tree = ET.parse(xml_file)
        root = tree.getroot()   
        imNames = []
        imBoxes = []
        imClasses = []  
        
        #Get the image and pad by half the crop size
        img = io.imread('%s/images/%s' % (path, filename))
        pad = cSize // 2 + 5
        img = np.pad(img, pad, mode = 'edge')[:, :, pad : pad + 3]
        pImg = Image.fromarray(img)
        print(img.shape)
        for i, child in enumerate(root):
                if child.tag == 'object':
                        #Get labels
                        obtype = float(child[0].text)
                        xmin = float(child[1][0].text)
                        ymin = float(child[1][1].text)
                        xmax = float(child[1][2].text)
                        ymax = float(child[1][3].text)  
                        #Mean coords 
                        xmean = np.mean([xmin, xmax]) + pad
                        ymean = np.mean([ymin, ymax]) + pad
                        #crop image
                        #cImg = img[int(xmean) - cSize // 2 + 1 : int(xmean) + cSize // 2 + pad, 
                        #           int(ymean) - cSize // 2 + 1 : int(ymean) + cSize // 2 + pad]
                        cImg = pImg.crop([xmean - cSize // 2, ymean - cSize // 2, xmean + cSize // 2, ymean + cSize // 2])
                        #Save cropped image
                        #io.imsave('%s/processed/%s_%i.png' % (path, filename[ : -4], i), cImg)  
                        cImg.save('%s/processed/%s_%i.png' % (path, filename[ : -4], i))
                        #Append the filename and label to the list 
                        imNames.append('%s_%i.png' % (filename[ : -4], i))
                        imBoxes.append(np.array([xmin - xmean, ymin - ymean, xmax - xmean, ymax - ymean]))
                        imClasses.append(obtype)

        return imNames, imBoxes, imClasses
        
namelist = os.listdir(dat_path + '/images')

Names, Boxes, Classes = [], [], []

for name in namelist:
        print(name)
        t_Names, t_Boxes, t_Classes = process_dat(name, dat_path)
        Names.extend(t_Names)
        Boxes.extend(t_Boxes)
        Classes.extend(t_Classes)

labels = list(zip(Names, Boxes, Classes))

labels_df = pd.DataFrame(labels, index = None, columns = ['filename', 'bounding_box', 'category'])
labels_df.to_csv(dat_path + '/labels.csv', )