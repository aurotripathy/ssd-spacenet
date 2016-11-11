
# With a newer OpenCV version (I use 3.1.0) it works like this:

import cv2
import os
from pudb import set_trace


class stitchFrames(object):
    def __init__(self, in_dir, out_dir):
        self.in_dir = in_dir
        self.out_dir = out_dir

    def stitch(self):
        # initialize the FourCC, video writer, dimensions of the frame
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        writer = None
        (h, w) = (None, None)

        count=0
        frame = self.in_dir + 'frame{}.jpg'.format(count)
        count += 1
        image = cv2.imread(frame)
        if image is None: 
            print 'Cant find the zeroth frame...exiting.'
            return -1

        (h, w) = image.shape[:2]
        writer = cv2.VideoWriter(self.out_dir + 'video.mov', fourcc, 10, (w, h), True)
            
        while True:
            frame = self.in_dir + 'frame{}.jpg'.format(count)
            image = cv2.imread(frame)
            if image is not None:
                writer.write(image)
            else:
                break
            count += 1
            
        writer.release()
        return count

# set_trace()

# unit test

# in_dir = 'detect/uploads/video_build_folder/'
# out_dir = './'
# frame_prefix = 'frame'
# video_file = 'video.mov'
# sf = stitchFrames(in_dir, out_dir)
# print 'Number of frames stitched-together {}'.format(sf.stitch())


