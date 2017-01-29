# converting the curl command below into code.
# curl -H "Content-type: multipart/form-data" 
#      -X POST https://shatterline.ngrok.io/imageClassPlusBB/ 
#      -F "file=@/home/tempuser/Pictures/pup_litter.jpg"
import cv2
import json
from pudb import set_trace

def execute_rest_api(img_name, outdir_path):
    url = "https://shatterline.ngrok.io/imageClassPlusBB/"
    # img_name = '/home/tempuser/Pictures/two_pups.jpg'
    files = {'file':(img_name,  open(img_name, 'rb'), "Content-type: multipart/form-data")}
    import requests
    with open(img_name, 'rb') as f:
        r = requests.post(url, files=files)
    print(r.status_code, r.reason)
    print r.text
    s = r.text

    import ast
    temp = ast.literal_eval(s)
    bb_list = ast.literal_eval(temp)

    image = cv2.imread(img_name)
    print 'image size {}'.format(image.shape)
    cpimg = image.copy()

    COLORS = ((0,0,0), (51, 51, 255), (255, 51, 51), (51, 255, 51), (255,255,0), 
              (0,255,255), (0,127,255), (128,0,255), (102,102,255), (255,102,102), (102,255,102) )
    tot_colors = len(COLORS)

    import random


    for bb in bb_list:
        print 'confidence: {}, (xmin, ymin): {}, (xmax, ymax): {}, label: {}'.format(bb['confidence'], 
                                                                                     bb['x_min_y_min'], 
                                                                                     bb['x_max_y_max'],
                                                                                     bb['label'])
        (xmin, ymin) = bb['x_min_y_min']
        (xmax, ymax) = bb['x_max_y_max']
        label = bb['label']
        color_index = random.randint(0,tot_colors)
        cv2.rectangle(cpimg, (xmin, ymin), (xmax, ymax), COLORS[color_index], 2)
        cv2.putText(cpimg, label, (xmin, ymin + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[color_index] , 1)

        output_img = 'pup_litter_with_bb.jpg'
        try:
            cv2.imwrite(output_img,cpimg)
        except ValueError:
            print "Something went wrong"


inputlist = open('../one-image.txt','r')
lines = inputlist.readlines()
for image_name in lines:
    image_name = image_name.replace('\n','')
    execute_rest_api(image_name, '.')

