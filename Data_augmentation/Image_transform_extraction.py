
import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
import glob
import datetime
import os
from pathlib import Path
from argparse import ArgumentParser

Folder_name =r"C:\Users\CHAMPION\Documents\INV\INV\Format 3"
Extension = ".JPEG"
'''
parser = ArgumentParser()
parser.add_argument("-f", "--files",dest="files", help="path to files ")

args = parser.parse_args()
'''
#Folder_name = args.files

#RESIZE
def add_light(image, gamma=1.0):
    print(image)
    print(filename)
    invGamma = 1.0/ gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name +"/"+ filename+ "_"+str(gamma)+Extension, image)
    else:
        cv2.imwrite(Folder_name +"/"+ filename+ "_"+str(gamma) + Extension , image)

def add_light_color(image, color, gamma=1.0):
    invGamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name + "/"+ filename+"_"+str(gamma)+Extension, image)
    else:
        cv2.imwrite(Folder_name +"/"+ filename+"_" +str(gamma) + Extension , image)


def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(Folder_name +"/"+ filename+"_"+datetime.datetime.now().strftime("%S")+ "_" + Extension, image)




images = []
for filenamee in os.listdir(Folder_name):
    filepath = os.path.join(Folder_name, filenamee)
    p = Path(filepath)
    filename=(p.stem)
    
    image = cv2.imread(os.path.join(Folder_name,filenamee))
  
    if image is not None:
        images.append(image)
        
        
        add_light(image, 0.8)
        add_light(image, 2.0)
        add_light(image, 0.3)        
        add_light(image, 2.3)
        add_light(image, 2.8)
        add_light(image, 2.2)        
        add_light(image, 1.0)
        add_light(image, 1.5)        
        add_light(image, 0.7)
        add_light(image, 0.4)
        add_light(image, 0.8)
        add_light(image, 0.6)
        add_light(image, 0.9)
        add_light(image, 0.5)    
        add_light(image, 0.38)
        add_light(image, 0.56)
        add_light(image, 0.2)
        add_light(image, 0.1)
        sharpen_image(image)

       


