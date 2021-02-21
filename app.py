from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
#import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import array

import json

# Keras
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

#dogbreed header
global labels

#new app
import base64
from PIL import Image
from scipy.misc import imsave, imread, imresize
from prepare_data import normalize
from keras.models import load_model
# Define a flask app
app = Flask(__name__)


def model_predict_cotton(img_path):

    # Model saved with Keras model.save()
    MODEL_PATH ='rps.h5'

    # Load your trained model
    model = load_model(MODEL_PATH)
    
    img = image.load_img(img_path, target_size=(150, 150))

    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images)

    if classes[0]>0:
        preds="The Leaf is not infected "
    else:
        preds="The Leaf is infected"
    
    return preds

def model_predict_malaria(img_path):

    # Model saved with Keras model.save()
    MODEL_PATH ='malaria_Transfer.h5'

    # Load your trained model
    model = load_model(MODEL_PATH)
    
    img = image.load_img(img_path, target_size=(150, 150))

    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images)

    if classes[0]>0:
        preds="The cell is not infected "
    else:
        preds="The cell is infected"
    
   
    
    return preds

def model_predict_facial(img_path):

    # Model saved with Keras model.save()
    MODEL_PATH ='face.h5'

    EMOTIONS_LIST = ["Angry", "Disgust","Happy", "Fear","Neutral", "Sad","Surprise"]

    # Load your trained model
    model = load_model(MODEL_PATH)
    
    img = image.load_img(img_path, target_size=(48, 48))

    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images)
    
    return EMOTIONS_LIST[np.argmax(classes)]
    

def model_predict_catdog(img_path):

    # Model saved with Keras model.save()
    MODEL_PATH ='dogbreedweights.inceptionv3.h5'

    LABELS_PATH = os.path.abspath('labels.txt')

    global labels

    with open(LABELS_PATH) as f:
        labels = f.readlines()

    labels = np.array([label.strip() for label in labels])

    # Load your trained model
    model = load_model(MODEL_PATH)
    
    img = image.load_img(img_path, target_size=(224, 224))

    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])

    preds=labels[np.argmax(model.predict(images))]
    
    return preds

def model_predict_trafic(img_path):

    # Model saved with Keras model.save()
    MODEL_PATH ='traffic_classifier.h5'

    classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }

    # Load your trained model
    model = load_model(MODEL_PATH)
    
    img = image.load_img(img_path, target_size=(30, 30))

    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])

    preds=classes[np.argmax(model.predict(images))+1]
    
    return preds



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/portfolio', methods=['GET'])
def portfolio():
    # Main page
    return render_template('index_port.html')

@app.route('/cotton', methods=['GET'])
def cotton():
    # Main page
    return render_template('index.html')

@app.route('/malaira', methods=['GET'])
def malaria():
    # Main page
    return render_template('indexmalaria.html')

@app.route('/facial_expression', methods=['GET'])
def facial():
    # Main page
    return render_template('indexfacial.html')

@app.route('/catdog', methods=['GET'])
def catdog():
    # Main page
    return render_template('indexcatdog.html')

@app.route('/sentiment', methods=['GET'])
def sentiment():
    # Main page
    return render_template('setiment.html')

@app.route('/traffic', methods=['GET'])
def traffic():
    # Main page
    return render_template('indextrafic.html')



@app.route('/predictcotton', methods=['GET', 'POST'])
def upload_cotton():
    if request.method == 'POST':
        # Get the file from post request
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict_cotton(file_path)
        result=preds
         
    
        return result
    return None


@app.route('/predictmalaria', methods=['GET', 'POST'])

def upload_malaria():
    if request.method == 'POST':
        # Get the file from post request
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict_malaria(file_path)
        result=preds

        
        return result
    return None

@app.route('/predictfacial', methods=['GET', 'POST'])
def upload_facial():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict_facial(file_path)
        result=preds
         
    
        return result
    return None   

@app.route('/predictcatdog', methods=['GET', 'POST'])
def upload_catdog():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict_catdog(file_path)
        result=preds
         
    
        return result
    return None   

@app.route('/predicttraffic', methods=['GET', 'POST'])
def upload_traffic():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict_trafic(file_path)
        result=preds
         
    
        return result
    return None   


@app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
def sent_anly_prediction():
    if request.method=='POST':
        text = request.form['text']
        model=keras.models.load_model('sarcasm.h5')
        
        with open("sarcasm.json", 'r') as f:
            datastore = json.load(f)
            sentences = []
            labels = []

        for item in datastore:
            sentences.append(item['headline'])
            labels.append(item['is_sarcastic'])

        vocab_size = 10000
        embedding_dim = 16
        max_length = 100
        trunc_type='post'
        padding_type='post'
        oov_tok = "<OOV>"
        training_size = 20000

        training_sentences = sentences[0:training_size]
        testing_sentences = sentences[training_size:]
        training_labels = labels[0:training_size]
        testing_labels = labels[training_size:]  

        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(training_sentences)

        word_index = tokenizer.word_index

        training_sequences = tokenizer.texts_to_sequences(training_sentences)
        training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
        testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)  

        sentence = [text]
        
        sequences = tokenizer.texts_to_sequences(sentence)
        padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        predict=model.predict(padded)
        classes=model.predict_classes(padded)

        if classes[0]==0:
            senti="not sarcastic"
            
        else:
             senti="sarcastic"
           

        
    return render_template('setiment.html', text=text, sentiment=senti, probability=predict[0]) 
#   return render_template('home.html', text=text, sentiment=sentiment, probability=probability, image=img_filename) 



@app.route("/doodle", methods=["GET", "POST"])
def ready():

    mlp = load_model("./models/mlp_94.h5")
    conv = load_model("./models/conv_95.5.h5")
    FRUITS = {0: "Apple", 1: "Banana", 2: "Grape", 3: "Pineapple"}
    if request.method == "GET":
        return render_template("index1.html")
    if request.method == "POST":
        data = request.form["payload"].split(",")[1]
        net = request.form["net"]

        img = base64.b64decode(data)
        with open('temp.png', 'wb') as output:
            output.write(img)
        x = imread('temp.png', mode='L')
        # resize input image to 28x28
        x = imresize(x, (28, 28))

        if net == "MLP":
            model = mlp
            # invert the colors
            x = np.invert(x)
            # flatten the matrix
            x = x.flatten()

            # brighten the image a bit (by 60%)
            for i in range(len(x)):
                if x[i] > 50:
                    x[i] = min(255, x[i] + x[i] * 0.60)

        if net == "ConvNet":
            model = conv
            x = np.expand_dims(x, axis=0)
            x = np.reshape(x, (28, 28, 1))
            # invert the colors
            x = np.invert(x)
            # brighten the image by 60%
            for i in range(len(x)):
                for j in range(len(x)):
                    if x[i][j] > 50:
                        x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)

        # normalize the values between -1 and 1
        x = normalize(x)
        val = model.predict(np.array([x]))
        pred = FRUITS[np.argmax(val)]
        classes = ["Apple", "Banana", "Grape", "Pineapple"]
        print (pred)
        print (list(val[0]))
        return render_template("index1.html", preds=list(val[0]), classes=json.dumps(classes), chart=True, putback=request.form["payload"], net=net)






if __name__ == '__main__':
    app.run(debug=True)


