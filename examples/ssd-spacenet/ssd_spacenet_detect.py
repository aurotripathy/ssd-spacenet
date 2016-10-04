# Converted from the ipyton notebook for SSD detection

# Section 1
# First, Load necessary libs and set up caffe and caffe_root

import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
from read_annotations import read_annotations
from pudb import set_trace

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '.'  # this file is expected to be in {caffe_root}/examples/ssd-spacenet
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

# Load LabelMap.
#---------------
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

conf_threshold = 0.25

# load Spacenet labels
labelmap_file = 'data/spacenet/labelmap_spacenet.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


# Load the net in the test phase for inference, and configure input preprocessing.
#----------------------------------------------------------------------------------

model_def = 'models/VGGNet/spacenet/SSD_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/spacenet/SSD_300x300/spacenet_SSD_300x300_iter_16696.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# Section2
# SSD based detection
#---------------------
# set net to batch size of 1

def run_detect_net(image, gt_list):
    # Run the net and examine the top-k results
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than a threshold.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_threshold]


    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]


    # Plot the boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(image)
    currentAxis = plt.gca()

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%.2f'%(score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

    # plot the ground_truth
    for gt in gt_list:
        xmin, xmax, ymin, ymax = gt
        # xmin = int(round(top_xmin[i] * image.shape[1]))
        # ymin = int(round(top_ymin[i] * image.shape[0]))
        # xmax = int(round(top_xmax[i] * image.shape[1]))
        # ymax = int(round(top_ymax[i] * image.shape[0]))
        # score = top_conf[i]
        # label = int(top_label_indices[i])
        # label_name = top_labels[i]
        display_txt = ''
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = (0, 0, 0.5)
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})


image_resize = 300
net.blobs['data'].reshape(1, 3, image_resize, image_resize)

with open('data/spacenet/test.txt') as f:
    for line in f.readlines():

        img_file = line.split(' ')[0]
        gt_file = line.split(' ')[1].split('\n')[0]

        image = caffe.io.load_image(expanduser('~') + 
                                    '/spacenet-data/' + img_file)
        # set_trace()
        bounding_box_gt_list = read_annotations(expanduser('~') + '/spacenet-data/' + gt_file)
        run_detect_net(image, bounding_box_gt_list)
        plt.show()


