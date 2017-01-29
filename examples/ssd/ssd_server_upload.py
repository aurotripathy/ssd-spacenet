import os
from flask import Flask, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from ssd_server_detect import SsdDetectionServer
from ssd_server_detect import SsdDetectionServerV2

from twilio.rest import Client

import ImageDraw
import Image

from flask import Response
# from flask import json

import json
import yaml

import cv2

from stitch_frames_to_video import stitchFrames

from pudb import set_trace
 
TRAINED_SZ_SQ = 500
UPLOAD_FOLDER = './detect/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_VID_EXTENSIONS = set(['mov'])
HTTP_SERVER_URL = 'https://8cc4c241.ngrok.io/detect/uploads/play.html'

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# labelmap_file = '../../data/VOC0712/labelmap_voc.prototxt' 
# model_def = '../../models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
# model_weights = '../../models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'
if TRAINED_SZ_SQ == 300:
    labelmap_file = '../../data/coco/labelmap_coco.prototxt' 
    model_def = '../../models/VGGNet/coco/SSD_300x300/deploy.prototxt'
    model_weights = '../../models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_240000.caffemodel'
elif TRAINED_SZ_SQ == 500:
    labelmap_file = '../../data/coco/labelmap_coco.prototxt' 
    model_def = '../../models/VGGNet/coco/SSD_500x500/deploy.prototxt'
    model_weights = '../../models/VGGNet/coco/SSD_500x500/VGG_coco_SSD_500x500_iter_200000.caffemodel'
else:
    print 'Error in image size; must be either 300 or 500'
    exit(2)


ssd_server_detect = SsdDetectionServer(labelmap_file, model_def, model_weights, TRAINED_SZ_SQ)
ssd_server_detect_v2 = SsdDetectionServerV2(labelmap_file, model_def, model_weights, TRAINED_SZ_SQ)

recipient_phone_number = 14088025434
# recipient_phone_number = '+919443845253'

def _send_sms_notification(to, message_body, callback_url):
    # Ensure that the env has the vaiables below defined
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID', None)
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN', None)
    twilio_number = os.environ.get('TWILIO_NUMBER', None) # TODO make this as input
    client = Client(account_sid, auth_token)
    client.messages.create(to=to,
                           from_=twilio_number,
                           body=message_body,
                           status_callback=callback_url)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_VID_EXTENSIONS



@app.route('/', methods=['GET', 'POST'])
def non_func_homepage():
    return '<p>No API calls at home</p>', 400

# route to return just a list of bounding boxes
@app.route('/imageClassPlusBB/', methods=['POST'])
def detect_image_class_plus_bb():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print 'file name {}'.format(file)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'orig_'+ filename))
            image = ssd_server_detect.load_image(UPLOAD_FOLDER + '/' + 'orig_' + filename)
            print 'The original dimensions are {}'.format(image.shape)
            bb_plus_class = ssd_server_detect_v2.run_detect_net_v2(image)            
            bb_plus_class = '{}'.format(bb_plus_class)

    # return jsonify(results=bb_plus_class)
    return Response(json.dumps(bb_plus_class),  mimetype='application/json')



@app.route('/detect/', methods=['GET', 'POST'])
def detect_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print 'file name {}'.format(file)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print 'I believe we made it this far...'
            filename = secure_filename(file.filename)
            print '... and filename is {}'.format(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image = ssd_server_detect.load_image(UPLOAD_FOLDER + '/' + filename)
            top_conf, top_label_indices, top_labels, \
            top_xmin, top_ymin, top_xmax, top_ymax = ssd_server_detect.run_detect_net(image)

            overlayed_file = UPLOAD_FOLDER + '/' + filename
            ssd_server_detect.plot_boxes(overlayed_file, image,  
                                         top_conf, top_label_indices, top_labels,                           
                                         top_xmin, top_ymin, top_xmax, top_ymax)     

            callback_url = request.base_url + 'notification/status/update'
            print 'call back url {}'.format(callback_url)
            message = 'Results' + ' ' + request.base_url.replace('/detect', '') + 'uploads/' + filename 
            _send_sms_notification(recipient_phone_number,
                                   message,
                                   callback_url)

            return redirect(url_for('detected_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload an Image you want Detected</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

from scale import scale_box
def scale_boxes(t_xmin, t_ymin, t_xmax, t_ymax, x_ratio, y_ratio):
    xmin, ymin, xmax, ymax = [], [], [], []
    for xmn, ymn, xmx, ymx in zip(t_xmin, t_ymin, t_xmax, t_ymax):
        new_box = scale_box(xmn, ymn, xmx, ymx, x_ratio, y_ratio)
        xmin.append(new_box[0])
        ymin.append(new_box[1])
        xmax.append(new_box[2])
        ymax.append(new_box[3])

    return xmin, ymin, xmax, ymax

@app.route('/detectimage/', methods=['POST'])
def detect_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print 'file name {}'.format(file)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'orig_'+ filename))
            image = ssd_server_detect.load_image(UPLOAD_FOLDER + '/' + 'orig_' + filename)
            print 'The original dimensions are {}'.format(image.shape)
            # resize to TRAINED_SZ_SQ x TRAINED_SZ_SQ since the model is for those dimensions
            image_resized = ssd_server_detect.resize_image(image, TRAINED_SZ_SQ) #caffe api, changed from cv2 to match load
            top_conf, top_label_indices, top_labels, \
            top_xmin, top_ymin, top_xmax, top_ymax = ssd_server_detect.run_detect_net(image_resized)

            x_fact = float(image.shape[0])/float(TRAINED_SZ_SQ)
            y_fact = float(image.shape[1])/float(TRAINED_SZ_SQ)
            print 'x_factor {:06.2f}, y_factor {:06.2f}'.format(x_fact, y_fact)
            scaled_top_xmin, scaled_top_ymin, scaled_top_xmax, scaled_top_ymax = scale_boxes(top_xmin, top_ymin, top_xmax, top_ymax, 
                                                                                       x_fact, y_fact) 

            results_string = ''
            for l, c in zip (top_labels, top_conf):
                results_string += '{} ({:04.2f}),'.format(l, c)
            overlayed_file = UPLOAD_FOLDER + '/' + filename
            print 'Resized image going to plotting {}'.format(image_resized.shape)
            # ssd_server_detect.plot_boxes(overlayed_file, image_resized,  
            #                              top_conf, top_label_indices, top_labels,                           
            #                              top_xmin, top_ymin, top_xmax, top_ymax)     

            ssd_server_detect.plot_boxes(overlayed_file, image,  
                                         top_conf, top_label_indices, top_labels,                           
                                         scaled_top_xmin, scaled_top_ymin, scaled_top_xmax, scaled_top_ymax) 
            

            callback_url = request.base_url + 'notification/status/update'
            print 'call back url {}'.format(callback_url)
            message = 'Detected:' + ' ' + results_string + ' ' + request.base_url.replace('/detectimage', '') + 'uploads/' + filename 
            _send_sms_notification(recipient_phone_number,
                                   message,
                                   callback_url)

    return "<p>Done!</p>"


from flask import send_from_directory

@app.route('/uploads/<filename>')
def detected_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/detect/notification/status/update', methods=["POST"])
def notification_delivery_status():
    print "###Delivered  the notification"
    return '''                                                                                                
    <!doctype html> 
    <title>Done</title>                                                                                
    <h1>Done!</h1>                                                                    
    '''


@app.route('/detectimage/notification/status/update', methods=["POST"])
def detect_image_notification_delivery_status():
    print "###Delivered  the notification"
    return '''                                                                                                
    <!doctype html> 
    <title>Done</title>                                                                                
    <h1>Done!</h1>                                                                    
    '''




# responds to a curl message like this:
# curl -H "Content-type: application/json" -X POST http://127.0.0.1:5000/phone -d '{"phone":"14088025434"}'
@app.route('/phone/', methods = ['POST'])
def api_phone():

    global recipient_phone_number
    if request.headers['Content-Type'] == 'text/plain':
        return "Text Message: " + request.data

    elif request.headers['Content-Type'] == 'application/json':
        print 'value of request.json {}'.format(json.dumps(request.json))
        # parsed_json = json.load(request.json) //unicode issue to be understood
        parsed_json = yaml.safe_load(json.dumps(request.json))
        recipient_phone_number = parsed_json['phone']
        print 'New recipient phone number {}'.format(recipient_phone_number)
        return "JSON Message: " + json.dumps(request.json)

    elif request.headers['Content-Type'] == 'application/octet-stream':
        f = open('./binary', 'wb')
        f.write(request.data)
        f.close()
        return "Binary message written!"

    else:
        return "415 Unsupported Media Type ;)"


import tempfile
import shutil
# http://stackoverflow.com/questions/27135470/python-how-to-create-a-directory-and-overwrite-an-existing-one-if-necessary
def create_fresh_build_folder(dir_name):

    if (os.path.exists(dir_name)):
        # `tempfile.mktemp` Returns an absolute pathname of a file that 
        # did not exist at the time the call is made. We pass
        # dir=os.path.dirname(dir_name) here to ensure we will move
        # to the same filesystem. Otherwise, shutil.copy2 will be used
        # internally and the problem remains.
        tmp = tempfile.mktemp(dir=os.path.dirname(dir_name))
        # Rename the dir.
        shutil.move(dir_name, tmp)
        # And delete it.
        shutil.rmtree(tmp)


        # At this point, even if tmp is still being deleted,
        # there is no name collision.
        os.makedirs(dir_name)


@app.route('/detectvid/', methods=['POST'])
def detect_vidcurl_syntax():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print 'file name {}'.format(file)
        if file and allowed_video_file(file.filename):
            print 'I believe we made it this far...'
            filename = secure_filename(file.filename)
            print '... and filename is {}'.format(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #  need to change processing from here onwards relative to photos
            vidcap = cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            outfile = os.path.join(app.config['UPLOAD_FOLDER'], 'detect_' + filename)
            length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            print 'Number of frames in file {}'.format(length)

            # set_trace()
            video_build_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'video_build_folder')
            # if not os.path.exists(os.path.dirname(video_build_folder)):
            #     try:
            #         os.makedirs(os.path.dirname(video_build_folder))
            #     except OSError as exc: # Guard against race condition
            #         if exc.errno != errno.EEXIST:
            #             raise
            create_fresh_build_folder(video_build_folder)
            results_set = set() # set of objects found
            for count in range(length):
                success, image = vidcap.read()
                if success:
                    # change into caffe format
                    image = ssd_server_detect.cv_to_caffe(image)
                    image_resized = ssd_server_detect.resize_image(image, TRAINED_SZ_SQ) #caffe api, changed from cv2 to match load
                    top_conf, top_label_indices, top_labels, \
                        top_xmin, top_ymin, top_xmax, top_ymax = ssd_server_detect.run_detect_net(image_resized)
            
                    # print 'Processing frame {}; Label Count {}'.format(count, len(top_labels))
                    for l, c in zip (top_labels, top_conf):
                        results_set.add(l)  # TODO add the confidence parameter later
                    overlayed_file = video_build_folder + '/' + 'frame{}.jpg'.format(count)
                    ssd_server_detect.plot_boxes(overlayed_file, image_resized,
                                                 top_conf, top_label_indices, top_labels,
                                                 top_xmin, top_ymin, top_xmax, top_ymax)
            print results_set

            # now stitch the jpeg frames together to generate a mov  vodes video
            stitched_frames = stitchFrames(video_build_folder + '/', app.config['UPLOAD_FOLDER'] + '/')
            count = stitched_frames.stitch()
            print 'Number of frames stitched {}'.format(count)

            callback_url = request.base_url + 'notification/status/update'
            print 'call back url {}'.format(callback_url)
            results_str = ', '.join(str(s) for s in results_set)
            # message = 'Detected:' + ' ' + results_str + ' ' + request.base_url.replace('/vidcurl', '') + 'uploads/' + 'Uploaded_vid.mov'
            message = 'Detected:' + ' ' + results_str + ' ' + HTTP_SERVER_URL
            _send_sms_notification(recipient_phone_number,
                                   message,
                                   callback_url)


    return "<p>Done!</p>"




@app.route('/detectvid/notification/status/update', methods=["POST"])
def detect_vid_notification_delivery_status():
    print "###Delivered  the video complete notification"
    return '''                                                                                                
    <!doctype html> 
    <title>Done</title>                                                                                
    <h1>Done!</h1>                                                                    
    '''

if __name__ == "__main__":
    app.run()
