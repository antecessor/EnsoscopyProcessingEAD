import os

import cv2
import numpy as np

from LoadingUtils import readImageAndLabel, showImageAndLabels, read_obj_names

bboxfolder = ''.join("./EAD2020_dataType_framesOnly/gt_bbox/")
imagefolder = ''.join("./EAD2020_dataType_framesOnly/frames/")
classfile = ''.join("./EAD2020_dataType_framesOnly/class_list.txt")

imageNames = os.listdir(imagefolder)
boxNames = os.listdir(bboxfolder)


def getClasses():
    return read_obj_names(classfile)


def getImageMaskLabelAndClasses(i, show=False):
    imageName = imagefolder + imageNames[i]
    boxName = bboxfolder + boxNames[i]
    image, bbox = readImageAndLabel(imageName, boxName, show=False)
    image = cv2.resize(image, (512, 512))
    w, h, channel = image.shape
    mask_img = np.zeros((w, h), dtype=int)
    for b in bbox:
        cls, x1, y1, x2, y2 = b
        mask_img[int(x1):int(x2), int(y1):int(y2)] = int(cls)
    if show == True:
        showImageAndLabels(bbox, [1, 2, 3, 4, 5, 6], mask_img)
    return image, bbox, mask_img


def getLen():
    return len(os.listdir(imagefolder))
