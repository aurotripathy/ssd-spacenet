#!/usr/bin/env python

import fnmatch
import os
from random import shuffle
from sklearn.cross_validation import train_test_split
from os.path import basename
from pudb import set_trace

annotation_path = '/home/tempuser/spacenet-data/Annotations/'
extensions = ['*.xml']
annotation_files = []
for root, dirnames, filenames in os.walk(annotation_path):
    for extension in extensions:
        for filename in fnmatch.filter(filenames, extension):
            annotation_files.append('Annotations' + '/' + filename)

print "Total number of annotation files, {}\n".format(len(annotation_files))

shuffle(annotation_files) # in-place

images_path = '/home/tempuser/spacenet-data/3band/'
extensions = ['*.tif']
img_files = []
for root, dirnames, filenames in os.walk(images_path):
    for extension in extensions:
        for filename in fnmatch.filter(filenames, extension):
            img_files.append('3band' + '/' + filename)

print "Total number of image files, {}\n".format(len(img_files))


# match them up as a validation

combined_list = []
for f in annotation_files:
    annot_img = os.path.splitext(basename(f))[0] 
    matches = [img for img in img_files if annot_img in img] 
    if matches:
        # print '{} {}'.format(matches[0], f)
        combined_list.append('{} {}'.format(matches[0], f))

# split the list into test and train
train_set, val_set = train_test_split(combined_list, test_size = 0.2)
print 'Split into train/val sets of length {}/{}'.format(len(train_set), len(val_set))
with open('trainval.txt', 'w') as f:
    for i in train_set:
        f.write('{}\n'.format(i))
with open('test.txt', 'w') as f:
    for i in val_set:
        f.write('{}\n'.format(i))


import cv2
import os.path
import numpy as np

# create test_name_size.txt from test.txt

with open('test.txt', 'r') as f:
    lines = f.readlines()
print 'Creating the test_name_size.txt file with {} lines...'.format(len(lines))

with open('test_name_size.txt', 'w') as f:
    for line in lines:
        img = line.split(' ')[0]
        im = cv2.imread('/home/tempuser/spacenet-data/' + img) 
        width, height = im.shape[:2]
        f.write('{} {} {}\n'.format(os.path.splitext(os.path.basename(img))[0], width, height))
print 'Done'
