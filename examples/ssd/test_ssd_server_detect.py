# !/usr/bin/env python
from ssd_server_detect import SsdDetectionServer
from os.path import expanduser
from read_annotations import read_annotations
import os
from pudb import set_trace
import numpy as np
import matplotlib.pyplot as plt

# Unit test

set_trace()

model_def = 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'

# load VOC labels for the 21 classes
labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'

ssd_server_detect = SsdDetectionServer(labelmap_file, model_def, model_weights)

#use data/VOC0712/test.txt if you want to test the whole set
with open('unit-test.txt') as f:
    for line in f.readlines():
        print 'processing file {}'.format(line)
        img_file = line.split(' ')[0]
        gt_file = line.split(' ')[1].split('\n')[0]

        image = ssd_server_detect.load_image(expanduser('~') + 
                                    '/data/VOCdevkit/' + img_file)
        # set_trace()
        bounding_box_gt_list = read_annotations(expanduser('~') + '/data/VOCdevkit/' + gt_file)
        top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax = ssd_server_detect.run_detect_net(image)
        ssd_server_detect.plot_boxes(os.path.basename(img_file), image, 
                                     top_conf, top_label_indices, top_labels, 
                                     top_xmin, top_ymin, top_xmax, top_ymax)
