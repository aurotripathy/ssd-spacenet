# converting the curl command below into code.
# curl -H "Content-type: multipart/form-data" 
#      -X POST https://shatterline.ngrok.io/imageClassPlusBB/ 
#      -F "file=@/home/tempuser/Pictures/pup_litter.jpg"

import json
from pudb import set_trace
url = "https://shatterline.ngrok.io/imageClassPlusBB/"
pic = '/home/tempuser/Pictures/pup_litter.jpg'
files = {'file':(pic,  open(pic, 'rb'), "Content-type: multipart/form-data")}
import requests
with open(pic, 'rb') as f:
    r = requests.post(url, files=files)
print(r.status_code, r.reason)
print r.text
s = r.text

import ast
temp = ast.literal_eval(s)
bb_list = ast.literal_eval(temp)

for bb in bb_list:
    print 'confidence: {}, (xmin, ymin): {}, (xmax, ymax): {}, label: {}'.format(bb['confidence'], 
                                                                                 bb['x_min_y_min'], 
                                                                                 bb['x_max_y_max'],
                                                                                 bb['label'])





