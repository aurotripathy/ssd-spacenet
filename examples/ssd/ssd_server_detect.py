# Refactored from the ipython notebook for SSD detection

import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
from read_annotations import read_annotations
from pudb import set_trace

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

import os
import sys

def _get_labelnames(labelmap, labels):
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

def get_labelname_V2(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    classindex = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                classindex.append(labelmap.item[i].label)
                break
        assert found == True
    return labelnames, classindex



class SsdDetectionServer(object):
    def __init__(self, labelmap_file, model_def, model_weights, size, threshold=0.4):

        plt.rcParams['figure.figsize'] = (10, 10)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

        # Make sure that caffe is on the python path:
        caffe_root = '.'  # this file is expected to be in {caffe_root}/examples/ssd
        os.chdir(caffe_root)
        sys.path.insert(0, 'python')

        self.conf_threshold = threshold

        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
        
        self.net.blobs['data'].reshape(1, 3, size, size)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104,117,123])) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    def run_detect_net(self, image):
        # Run the net and examine the top-k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than a threshold.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.conf_threshold]


        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = _get_labelnames(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        return top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax
    
    def plot_boxes(self, save_in_file, image, 
                   top_conf, top_label_indices, top_labels, 
                   top_xmin, top_ymin, top_xmax, top_ymax):

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        plt.imshow(image)
        currentAxis = plt.gca()

        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top='off', bottom='off',      # ticks along the bottom edge are off
            left='off', right='off',
            labelbottom='off', labelleft='off')      # ticks along the bottom edge are off



        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            display_txt = '%s %.2f'%(label_name, score)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label % len(colors)]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        plt.savefig(save_in_file, bbox_inches='tight')
        plt.hold(False)

    def plot_boxes_into_image(self, image, 
                   top_conf, top_label_indices, top_labels, 
                   top_xmin, top_ymin, top_xmax, top_ymax):

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        plt.imshow(image)
        currentAxis = plt.gca()

        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top='off', bottom='off',      # ticks along the bottom edge are off
            left='off', right='off',
            labelbottom='off', labelleft='off')      # ticks along the bottom edge are off



        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            display_txt = '%s %.2f'%(label_name, score)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label % len(colors)]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        plt.hold(False)
        # Now we can save it to a numpy array.
        fig = plt.figure()
        fig.add_subplot(111)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def load_image(self, image_file):
        return caffe.io.load_image(image_file)

    def resize_image(self, image, size):
        return caffe.io.resize_image( image, (size, size), interp_order=3 )


    def cv_to_caffe(self, image):
        image = image / 255.
        return image[:,:,(2,1,0)]


COLORS = ((0,0,0), (51, 51, 255), (255, 51, 51), (51, 255, 51), (255,255,0),
          (0,255,255), (0,127,255), (128,0,255), (102,102,255), (255,102,102), (102,255,102) )
tot_colors = len(COLORS)

import cv2
from os import path
import skimage
import skimage.io as skio
import json

class SsdDetectionServerV2(object):
    def __init__(self, labelmap_file, model_def, model_weights, size, threshold=0.4):


        # Make sure that caffe is on the python path:
        caffe_root = '.'  # this file is expected to be in {caffe_root}/examples/ssd
        os.chdir(caffe_root)
        sys.path.insert(0, 'python')

        self.conf_threshold = threshold

        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
        
        self.net.blobs['data'].reshape(1, 3, size, size)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104,117,123])) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    def run_detect_net_v2(self, img):
        imagename = 'tempimage.jpg'
        # image = cv2.imread(imgpath)
        self.cpimg = img.copy()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = skimage.img_as_float(image).astype(np.float32)

        # Run the net and examine the top-k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than a threshold.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.conf_threshold]


        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels, top_class_index = get_labelname_V2(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        bb_plus_class_list = []
        if top_conf.shape[0] > 0:
            for i in xrange(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * image.shape[1]))
                ymin = int(round(top_ymin[i] * image.shape[0]))
                xmax = int(round(top_xmax[i] * image.shape[1]))
                ymax = int(round(top_ymax[i] * image.shape[0]))
                score = top_conf[i]
                label = top_labels[i]
                color_index = top_class_index[i] % tot_colors
                name = '%s: %.2f'%(label, score)
                # print 'label: {} (xmin, ymin) = ({}, {}), (xmax, ymax) = ({}, {}) color_index {}'.format(label, xmin, ymin, xmax, ymax, color_index)                                          
                cv2.rectangle(self.cpimg, (xmin, ymin), (xmax, ymax), COLORS[color_index], 2)
                cv2.putText(self.cpimg, name, (xmin, ymin + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[color_index] , 1)
                bb_plus_class_list.append({"label":str(label), "confidence": score, 
                                      "x_min_y_min":(xmin, ymin), "x_max_y_max":(xmax, ymax)})

            output_img = path.join('outdir', imagename)
            print 'Outputting image at {}'.format(output_img)
            try:
                cv2.imwrite(output_img, self.cpimg)
            except ValueError:
                print "Something went wrong"

        else:
            output_img = path.join('outdir', imagename)
            print output_img
            cv2.imwrite(output_img, self.cpimg)

        return bb_plus_class_list

    def load_image_v2(self, image_file):
        return caffe.io.load_image(image_file)

    def resize_image_v2(self, image, size):
        return caffe.io.resize_image( image, (size, size), interp_order=3 )


    def cv_to_caffe(self, image):
        image = image / 255.
        return image[:,:,(2,1,0)]

    def superimpose_bb_label(self, imagename):
        pass



# # Unit test

# model_def = 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
# model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'

# # load VOC labels for the 21 classes
# labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'

# ssd_server_detect = SsdDetectionServer(labelmap_file, model_def, model_weights)

# with open('unit-test.txt') as f:
#     for line in f.readlines():

#         img_file = line.split(' ')[0]
#         gt_file = line.split(' ')[1].split('\n')[0]

#         image = ssd_server_detect.load_image(expanduser('~') + 
#                                     '/data/VOCdevkit/' + img_file)
#         # set_trace()
#         bounding_box_gt_list = read_annotations(expanduser('~') + '/data/VOCdevkit/' + gt_file)
#         top_conf, top_label_indices, top_labels, top_xmin, top_ymin, top_xmax, top_ymax = ssd_server_detect.run_detect_net(image)
#         ssd_server_detect.plot_boxes(os.path.basename(img_file), image, 
#                                      top_conf, top_label_indices, top_labels, 
#                                      top_xmin, top_ymin, top_xmax, top_ymax)



# # Unit test TEST THIS API FROM THE LOCAL FOLDER

# model_def = '../../models/VGGNet/coco/SSD_500x500/deploy.prototxt'
# model_weights = '../../models/VGGNet/coco/SSD_500x500/VGG_coco_SSD_500x500_iter_200000.caffemodel'

# # load VOC labels for the 21 classes
# labelmap_file = '../../data/coco/labelmap_coco.prototxt'

# TRAINED_SZ_SQ = 500
# ssd_server_detect_v2 = SsdDetectionServerV2(labelmap_file, model_def, model_weights, TRAINED_SZ_SQ)

# inputlist = open('frames.txt','r')
# lines = inputlist.readlines()
# for line in lines:
#     line = line.replace('\n','')
#     image = ssd_server_detect_v2.load_image_v2(line)
#     # image_resized = ssd_server_detect.resize_image_v2(image, TRAINED_SZ_SQ)
#     bb = ssd_server_detect_v2.run_detect_net_v2(image)
#     print 'json data {}'.format(bb)

