#! /usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pudb import set_trace
from os import listdir
from os.path import isfile, join
from argparse import ArgumentParser

def zero_intensity_percent(img_path):
    img = cv2.imread(img_path)
    color = ('r', 'b', 'g')
    zero_intensity_percent = [0,0,0]
    for i, c in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0,256])
        zero_intensity_percent[i] = histr[0]/sum(histr) * 100
        plt.plot(histr, color=c)
        plt.xlim([0,256])
    
    return np.mean(zero_intensity_percent)

def get_parameters():
    parser = ArgumentParser()
    parser.add_argument("-d", "--input_dir", dest="input_dir",
                         help="input directory containing the images", required=True)
    parser.add_argument("-t", "--zero-threshold", dest="z_threshold",
                         help="Threshold below which image will be accepted for training", 
                        required=True)
    args = parser.parse_args()
    return args.input_dir, args.z_threshold


in_dir, z_threshold = get_parameters()

f_list = [join(in_dir, f) for f in listdir(in_dir) if isfile(join(in_dir, f))]

for i, f in enumerate(f_list):
    print 'File:{}, Image # {}, Pitch dark: {}%'.format(f, i, zero_intensity_percent(f))




