#! /usr/bin/env python

''' Using  processedData which is already tiled and has the corresponding pixel coordinates (to sub-pixel accuracy):
3band.tar.gz: This compressed tar archive contains 7186 3-band GeoTIFF files. 
These files were created by cutting the the srcData above into 200m x 200m images.
'''

from spaceNet import geoTools as gT
import numpy as np
from pudb import set_trace
from shapely.geometry.polygon import LinearRing
import os
import shutil
import xml.etree.cElementTree as ET
import sys
from indent import indent
import cv2

json_file = '/home/tempuser/spacenet-data/vectorData/summaryData/AOI_1_Rio_polygons_solution_3Band.geojson'
folder_name = 'spacenet'

def bounding_rect(poly):
    n = np.shape(poly)[0]
    poly = poly.reshape(n,2)
    ring = LinearRing(poly)
    mn_mx_xy = ring.bounds
    if mn_mx_xy:
        (mn_x, mn_y, mx_x, mx_y) = mn_mx_xy
        b_box = np.asarray([[mn_x, mn_y], [mx_x, mn_y], 
                            [mx_x, mx_y], [mn_x, mx_y]], np.int32)
        w, h = mx_x - mn_x, mx_y - mn_y
        return True, b_box.reshape(4,1,2), w, h, w * h 
    else:
        return False, np.zeros((4,1,2)), 0, 0, 0


def bounding_rect_min_max(poly):
    n = np.shape(poly)[0]
    poly = poly.reshape(n,2)
    ring = LinearRing(poly)
    mn_mx_xy = ring.bounds
    if mn_mx_xy:
        (mn_x, mn_y, mx_x, mx_y) = mn_mx_xy
        w, h = mx_x - mn_x, mx_y - mn_y
        return True,  mn_mx_xy, w, h, w * h 
    else:
        return False, (0, 0, 0, 0), 0, 0, 0

if __name__ == "__main__":
    gt_truth = json_file

    polys_sol = gT.importgeojson(gt_truth, removeNoBuildings=True)

    polys_sol_mids_list = np.asarray([item['ImageId'] 
                                     for item in polys_sol if item["ImageId"] > 0 
                                     and item['BuildingId']!=-1])

    polys_sol_bids_list = np.asarray([item['BuildingId'] 
                                     for item in polys_sol if item["ImageId"] > 0 
                                     and item['BuildingId']!=-1])

    polys_sol_polys_list = np.asarray([item['poly'] 
                                      for item in polys_sol if item["ImageId"] > 0 
                                      and item['BuildingId']!=-1])

out_dir = '/home/tempuser/spacenet-data/Annotations'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

curr_im_id = ''
f = open(out_dir + '/' + 'empty.xml', 'w')
root = ET.Element('annotation')
tree = ET.ElementTree(root)
for im_id, b_id, geom in zip(polys_sol_mids_list, polys_sol_bids_list, 
                          polys_sol_polys_list):
    if curr_im_id != im_id:
        indent(root)
        #tree.write(sys.stdout)
        tree.write(f)
        f.close()
        curr_im_id = im_id
        f = open(out_dir + '/' + im_id + '.xml', 'w')

        root = ET.Element('annotation')
        tree = ET.ElementTree(root)
        folder_e = ET.SubElement(root, 'folder')
        folder_e.text = folder_name
        f_name_e = ET.SubElement(root, 'filename')
        f_name_e.text = im_id + '.tif'

        # set_trace()
        im = cv2.imread('/home/tempuser/spacenet-data/3band' + '/' + '3band_' + f_name_e.text)
        height, width, depth = im.shape # (width,height) tuple
        size_e = ET.SubElement(root, 'size')
        depth_e = ET.SubElement(size_e, 'depth')
        depth_e.text = '3'
        height_e = ET.SubElement(size_e, 'height')
        height_e.text = str(height)
        width_e = ET.SubElement(size_e, 'width')
        width_e.text = str(width)


    pts = geom.GetGeometryRef(0)
    print '--------{}----{}------\n'.format(im_id, b_id)
    # f.write('--------{}----{}------\n'.format(im_id, b_id))

    obj_e = ET.SubElement(root, 'object')
    name_e = ET.SubElement(obj_e, 'name')
    name_e.text = 'building'
    bndbox_e = ET.SubElement(obj_e, 'bndbox')

    poly_pts = []
    for p in range(pts.GetPointCount()):
        poly_pts.append([int(round(pts.GetX(p))), int(round(pts.GetY(p)))])
        print 'x {}, y {}'.format(poly_pts[-1][0], poly_pts[-1][1])
        # f.write('x {}, y {}\n'.format(poly_pts[-1][0], poly_pts[-1][1]))

    
    poly_pts = np.array(poly_pts[0:-1], np.int32)
    poly_pts = poly_pts.reshape((-1,1,2))
        
    ok, mnx_mny_mxx_mxy, width, height, area = bounding_rect_min_max(poly_pts)
    (mnx, mny, mxx, mxy) = mnx_mny_mxx_mxy 

    xmin_e = ET.SubElement(bndbox_e, 'xmin') 
    xmin_e.text = str(int(mnx))
    ymin_e = ET.SubElement(bndbox_e, 'ymin') 
    ymin_e.text = str(int(mny))
    xmax_e = ET.SubElement(bndbox_e, 'xmax') 
    xmax_e.text = str(int(mxx))
    ymax_e = ET.SubElement(bndbox_e, 'ymax') 
    ymax_e.text = str(int(mxy))


    print 'box params w:{}, h:{}, area:{}\n'.format(width, height, area)
    # f.write('box params w:{}, h:{}, area:{}\n'.format(width, height, area))

indent(root)
# tree.write(sys.stdout)
tree.write(f)
f.close()
os.remove(out_dir + '/' + 'empty.xml')
