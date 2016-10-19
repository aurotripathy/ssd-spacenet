import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from ssd_server_detect import SsdDetectionServer

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

labelmap_file = '../../data/VOC0712/labelmap_voc.prototxt' 
model_def = '../../models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
model_weights = '../../models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'
ssd_server_detect = SsdDetectionServer(labelmap_file, model_def, model_weights)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
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

from flask import send_from_directory

@app.route('/uploads/<filename>')
def detected_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

