# !/usr/bin/env python
from ssd_server_detect import SsdDetectionServer
from os.path import expanduser
from read_annotations import read_annotations
import os
from pudb import set_trace
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Unit test

parser = ArgumentParser()

# Add more options if you like
parser.add_argument("-m", "--model", dest="model",
                    help="pick from the coco or the VOC", required=True, choices=('coco', 'VOC'))

try:
    input = parser.parse_args()
    model = input.model
    print 'model is {}'.format(model)
except IOError, msg:
    parser.error(str(msg))

if model == 'VOC':
    model_def = 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
    model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'
    labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
elif model == 'coco':
    labelmap_file = 'data/coco/labelmap_coco.prototxt'
    model_def = 'models/VGGNet/coco/SSD_300x300/deploy.prototxt'
    model_weights = 'models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_240000.caffemodel'
else:
    print 'Unknown error'
    exit(2)

ssd_server_detect = SsdDetectionServer(labelmap_file, model_def, model_weights)

#use data/VOC0712/test.txt if you want to test the whole set
with open('unit-test.txt') as f:
    for line in f.readlines():
        print 'processing file {}'.format(line)
        img_file = line.split(' ')[0]
        gt_file = line.split(' ')[1].split('\n')[0]

        image = ssd_server_detect.load_image(expanduser('~') + 
                                    '/data/VOCdevkit/' + img_file)
        print 'Image shape {}'.format(image.shape)

        # set_trace()
        bounding_box_gt_list = read_annotations(expanduser('~') + '/data/VOCdevkit/' + gt_file)
        top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax = ssd_server_detect.run_detect_net(image)

        set_trace()

        for c, l in zip(top_conf, top_labels): print 'conf {:04.2f} label {}'.format(c, l)

        ssd_server_detect.plot_boxes(os.path.basename(img_file), image, 
                                     top_conf, top_label_indices, top_labels, 
                                     top_xmin, top_ymin, top_xmax, top_ymax)
