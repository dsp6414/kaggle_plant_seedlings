# imports
'''
splits out the green bits
https://www.kaggle.com/gaborvecsei/plant-seedlings-fun-with-computer-vision
see https://www.kaggle.com/omkarsabnis/seedling-classification-using-cnn-v13-0-95
'''
import os
import operator
from shutil import copyfile
import cv2
import numpy as np


def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

def modify_images_in_directory(dir):
    '''
    strip all unwanted content from files in dir
    :param dir: location of files to be manipulated
    :return:
    '''
    # get list of files in dir
    files = os.listdir(dir)
    for file in files:
        fq_file = os.path.join(dir, file)
        src = cv2.imread(fq_file, cv2.IMREAD_UNCHANGED)  # read a file

        image_segmented = segment_plant(src)
        image_sharpen = sharpen_image(image_segmented)

        cv2.imwrite(fq_file, image_sharpen)

#Add a ref to directory where train/test data is
data_dir = "/home/keith/data/plant_seedlings/"
data_dir_train = os.path.join(data_dir,'train')

#count how many files in each folder
allfiles = {}
for root, dirs, files in os.walk(data_dir_train):
    for dir in dirs:
        allfiles[dir] = len(os.listdir(os.path.join(root,dir)))

for key, value in allfiles.items():
    dirtrain = os.path.join(data_dir_train, key)
    modify_images_in_directory(dirtrain)

#mod the test images
data_dir_test = os.path.join(data_dir,'test')
modify_images_in_directory(data_dir_test)
