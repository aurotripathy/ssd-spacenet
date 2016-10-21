import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from ssd_server_detect import SsdDetectionServer

from twilio.rest import Client

UPLOAD_FOLDER = './detect/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

labelmap_file = '../../data/VOC0712/labelmap_voc.prototxt' 
model_def = '../../models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
model_weights = '../../models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'
ssd_server_detect = SsdDetectionServer(labelmap_file, model_def, model_weights)

recepient_phone_number = 14088025434
# recepient_phone_number = '+919443845253'

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

@app.route('/', methods=['GET', 'POST'])
def non_func_homepage():
    return '<p>No API calls at home</p>', 400

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
            _send_sms_notification(recepient_phone_number,
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


@app.route('/curl/', methods=['POST'])
def detect_curl_syntax():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print 'file name {}'.format(file)
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
            message = 'Results' + ' ' + request.base_url.replace('/curl', '') + 'uploads/' + filename 
            _send_sms_notification(recepient_phone_number,
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


@app.route('/curl/notification/status/update', methods=["POST"])
def curl_notification_delivery_status():
    print "###Delivered  the notification"
    return '''                                                                                                
    <!doctype html> 
    <title>Done</title>                                                                                
    <h1>Done!</h1>                                                                    
    '''
