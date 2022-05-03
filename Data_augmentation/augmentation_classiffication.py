import argparse
import warnings

from augmentation.augmentation import DatasetGenerator
from augmentation_config import *

import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
import glob
import datetime
import os
import shutil

from pathlib import Path


warnings.filterwarnings("ignore")

DEFAULT_DOWNLOAD_LIMIT = 50
DEFAULT_OUTPUT_FOLDER = "/output"






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    parser.add_argument('-folder',
                        help='Folder input path containing images that will be augmented',
                        required=True,
                        type=str
                        )

    parser.add_argument('-limit',
                        '-l',
                        help='Number of files to generate (default: %s)'
                             % DEFAULT_DOWNLOAD_LIMIT,
                        required=True,
                        type=int
                        )

    parser.add_argument('-dest',
                        '-d',
                        help='Folder destination for augmented image. (default: [folder input path] + %s)'
                        % DEFAULT_OUTPUT_FOLDER,
                        type=str,
                        default=None
                        )

    args = parser.parse_args()
    folder_path=args.folder
    #shutil.rmtree(folder_path+"/"+"tmp", ignore_errors=True)
    
    '''
    def add_light(image, gamma=1.0):
        invGamma = 1.0/ gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        image=cv2.LUT(image, table)
        if gamma>=1:
            cv2.imwrite(Folder_name + "/"+str(q)+ datetime.datetime.now().strftime("%Y%m%d_%H%M%S") +str(gamma)+Extension, image)
        else:
            cv2.imwrite(Folder_name + "/"+str(q) + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + str(gamma) + Extension , image)

    def add_light_color(image, color, gamma=1.0):
        invGamma = 1.0 / gamma
        image = (color - image)
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        image=cv2.LUT(image, table)
        if gamma>=1:
            cv2.imwrite(Folder_name + "/"+str(q)+ datetime.datetime.now().strftime("%Y%m%d_%H%M%S") +str(gamma)+Extension, image)
        else:
            cv2.imwrite(Folder_name + "/"+str(q) + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + str(gamma) + Extension , image)


    def sharpen_image(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        cv2.imwrite(Folder_name + "/"+str(q) + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + Extension, image)




    def contrast_image(image, contrast):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for
                          row in image[:, :, 2]]
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        cv2.imwrite(Folder_name +  "/"+str(q) + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + str(contrast) + Extension, image)

    def edge_detect_canny_image(image, th1, th2):
        image = cv2.Canny(image, th1, th2)
        cv2.imwrite(Folder_name+ "/"+str(q) + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") +str(th1) + "*" + str(th2) + Extension, image)


    def scale_image(image, fx, fy):
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(Folder_name + "/"+str(q)+ datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + str(fx) + str(fy) + Extension, image)

    def translation_image(image, x, y):
        #images=[]
        pic_num = 1
        rows, cols, c = image.shape
        M = np.float32([[1, 0, x/2], [0, 1, y/2]])
        image = cv2.warpAffine(image, M, (cols, rows))

        cv2.imwrite(Folder_name+ "/"+str(q)+ datetime.datetime.now().strftime("%Y%m%d_%H%M%S") +str(x)+str(y) +Extension, image)



        #images.append(translation_image)

    def rotate_image(image, deg):
        rows, cols, c = image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
        image = cv2.warpAffine(image, M, (cols, rows))
        cv2.imwrite(Folder_name + "/"+str(q) + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + str(deg) + Extension, image)

     
    if os.path.exists(folder_path+"/"+"tmp"):
        print("Found")
    
    else:
        os.mkdir(folder_path+"/"+"tmp")
    #RESIZE
    Folder_name=folder_path+"/"+'tmp'+filename
    p = Path(Folder_name)
    print("q",p.stem)
    q=p.stem
    Extension = ".JPEG"
    pathtmp=folder_path+"/"+"tmp"
    images = []
    for filename in os.listdir(folder_path):
        image = cv2.imread(os.path.join(folder_path,filename))
        if image is not None:
            images.append(image)
         
            add_light(image, 0.4)
            add_light(image, 2.0)
            add_light(image, 2.2)
            add_light(image, 1.0)
            add_light(image, 1.5)
            add_light(image, 0.6)
            add_light(image, 0.3)
            add_light(image, 0.5)
            sharpen_image(image)
    '''
    generator = DatasetGenerator(
            folder_path=args.folder,
            num_files=args.limit,
            folder_destination=args.folder + DEFAULT_OUTPUT_FOLDER if args.dest is None else args.dest)
    if 'rotate' in DEFAULT_OPERATIONS:
            generator.rotate(probability=DEFAULT_ROTATE_PROBABILITY,
                             max_left_degree=DEFAULT_ROTATE_MAX_LEFT_DEGREE,
                             max_right_degree=DEFAULT_ROTATE_MAX_RIGHT_DEGREE)

    if 'blur' in DEFAULT_OPERATIONS:
            generator.blur(probability=DEFAULT_BLUR_PROBABILITY)

    if 'random_noise' in DEFAULT_OPERATIONS:
            generator.random_noise(probability=DEFAULT_RANDOM_NOISE_PROBABILITY)
        
    generator.execute()
        
