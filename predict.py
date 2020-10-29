from __future__ import division, print_function, absolute_import
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
from collections import deque
from keras import backend
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
import PIL.ImageOps
import shutil
from PIL import Image


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def main(yolo):
    #while True:
    img = ('img/street.jpg')
    image = Image.open(img)
    #image = Image.fromarray(frame[...,::-1])
    #boxs, confidence, class_names = yolo.detect_image(image)
    r_image = yolo.detect_image(image)
    r_image.show()
    #print(boxs,confidence,class_names)
    yolo.close_session()
if __name__ == '__main__':
    main(YOLO())
