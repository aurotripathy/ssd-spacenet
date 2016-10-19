import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["building"]


def read_annotations(annot_file):
    in_file = open(annot_file)
    tree=ET.parse(annot_file)
    root = tree.getroot()

    bb_list = []
    for obj in root.iter('object'):
 
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), 
             int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text))
        bb_list.append(b)
    return bb_list


