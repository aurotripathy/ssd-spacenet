# http://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm#Combining

from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely import affinity


def scale_box(xmin, ymin, xmax, ymax, xfact, yfact):
    rect = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax,ymin)])
    rect_s = affinity.scale(rect, xfact=xfact, yfact=yfact, origin='center')
    # rect_s = affinity.scale(rect, xfact=xfact, yfact=yfact, origin=(0., 0.))
    # print rect_s.exterior.coords[:-1]
    for coord in rect_s.exterior.coords[:-1]: 
        print '{:06.2f} {:06.2f}'.format(coord[0], coord[1]) 
    return  rect_s.exterior.coords[0][0], rect_s.exterior.coords[0][1], rect_s.exterior.coords[2][0], rect_s.exterior.coords[2][1] 


