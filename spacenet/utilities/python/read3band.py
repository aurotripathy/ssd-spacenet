#! /usr/bin/env python

''' 
A simple program that ensures we can read the geojson data correctly
We use the  processedData folder which contains the pre-tiled and has the building contours in 
pixel coordinates (to sub-pixel accuracy):
3band.tar.gz: This compressed tar archive contains 7186 3-band GeoTIFF files. 
For a detailed explanation on the dataset, go to https://aws.amazon.com/public-data-sets/spacenet/
'''

from spaceNet import geoTools as gT
import numpy as np
1
if __name__ == "__main__":
    truth_fp = '/media/tempuser/RAID 5/spacenet/data/AOI_1_Rio/processedData/vectorData/summaryData/AOI_1_Rio_polygons_solution_3Band.geojson'

    sol_polys = gT.importgeojson(truth_fp, removeNoBuildings=True)

    sol_polysIdsList = np.asarray([item['ImageId'] for item in sol_polys if item["ImageId"] > 0 and \
                                   item['BuildingId']!=-1])

    sol_polysBIdsList = np.asarray([item['BuildingId'] for item in sol_polys if item["ImageId"] > 0 and \
                                    item['BuildingId']!=-1])

    sol_polysPoly = np.asarray([item['poly'] for item in sol_polys if item["ImageId"] > 0 and \
                                item['BuildingId']!=-1])


