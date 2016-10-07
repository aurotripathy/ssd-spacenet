#! /usr/bin/env python

''' Using  processedData which is already tiled and has the corresponding pixel coordinates (to sub-pixel accuracy):
3band.tar.gz: This compressed tar archive contains 7186 3-band GeoTIFF files. 
These files were created by cutting the the srcData above into 200m x 200m images.
'''

from spaceNet import geoTools as gT
import numpy as np
from pudb import set_trace
import cv2

from shapely.geometry.polygon import LinearRing

def bounding_rect(poly):
    # set_trace()
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

x_phy, y_phy = 200, 200
wnd_str = 'spacenet'
black = (0, 0, 0)
white = (255, 255, 255)
if __name__ == "__main__":
    truth_fp = '/media/tempuser/RAID 5/spacenet/data/AOI_1_Rio/processedData/vectorData/summaryData/2025.geojson'

    sol_polys = gT.importgeojson(truth_fp, removeNoBuildings=True)

    sol_polysIdsList = np.asarray([item['ImageId'] for item in sol_polys if item["ImageId"] > 0 and \
                                   item['BuildingId']!=-1])

    sol_polysBIdsList = np.asarray([item['BuildingId'] for item in sol_polys if item["ImageId"] > 0 and \
                                    item['BuildingId']!=-1])

    sol_polysPoly = np.asarray([item['poly'] for item in sol_polys if item["ImageId"] > 0 and \
                                item['BuildingId']!=-1])

img_path = '/media/tempuser/RAID 5/spacenet/data/AOI_1_Rio/processedData/3band/3band_013022223132_Public_img2025.tif'
img = cv2.imread(img_path)
x, y =  img.shape[0], img.shape[1]
print 'x {}, y {} dims'.format(x,y)
 
cv2.namedWindow(wnd_str,cv2.WINDOW_NORMAL)
cv2.moveWindow(wnd_str, 0, 0)

for ids, bId, geom in zip(sol_polysIdsList, sol_polysBIdsList, sol_polysPoly):
    pts = geom.GetGeometryRef(0)
    print '--------{}----{}------'.format(ids, bId)

    poly_pts = []
    for p in range(pts.GetPointCount()):
        poly_pts.append([int(round(pts.GetX(p))), int(round(pts.GetY(p)))])
        print 'x {}, y {}'.format(poly_pts[-1][0], poly_pts[-1][1])

    poly_pts = np.array(poly_pts[0:-1], np.int32)
    poly_pts = poly_pts.reshape((-1,1,2))
    cv2.polylines(img, [poly_pts], True, white, 1)

    ok, rect, width, height, area = bounding_rect(poly_pts)
    print 'bounding box params w:{}, h:{}, area:{}'.format(width, height, area)
    # set_trace()
    if ok:
        cv2.polylines(img, [rect], True, black, 1)

cv2.imshow(wnd_str,img)

cv2.waitKey(0)
cv2.destroyAllWindows()





