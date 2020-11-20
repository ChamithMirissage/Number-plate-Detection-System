from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
from PIL import Image
import io

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from tempfile import NamedTemporaryFile
from shutil import copyfileobj
from os import remove

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_file, Response, send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#from yolo_video1 import detect_img
import yolo_masks
from yolo_masks import YOLO, detect_video

# Define a flask app
app = Flask(__name__)

picFolder = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = picFolder

# Model saved with Keras model.save()
##MODEL_PATH1 = 'resnet50_best.h5'

# Load your trained model
#model1 = load_model(MODEL_PATH1)
#model1._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
#print('Model1 loaded. Check http://127.0.0.1:5000/')


def detect_img(image):
    while True:
        y = YOLO()
        r_image, w_m_persons = y.detect_image(image)
        #r_image.show()
        break
    #y.close_session()
    return w_m_persons

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        #file_path = os.path.join(
        #    basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        img = image.load_img(file_path, target_size=(416, 416))
		
        # Make prediction
        without_masks = detect_img(img)
        
        return str(without_masks)
    return None 

if __name__ == '__main__':
    app.run(debug=True)

