from flask import Flask, render_template, url_for, session, redirect, request
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired, NumberRange
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os




app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'

genders = ['Female', 'Male']
races = ['Asian', 'Black', 'White']

# Load Model
model_gender = load_model('./models/gender_08_0.215.h5', compile=False)
model_race = load_model('./models/race_05_0.281.h5', compile=False)
model_age = load_model('./models/age_77_77.805.h5', compile=False)


face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')


@app.route('/')
def index() :
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def uploadFile() :
    if request.method == 'POST' :
        f = request.files['file']

        # upload path
        f.save('./uploaded_images/' + secure_filename(f.filename))
        # print(f.filename)

        
        pic = cv2.imread('./uploaded_images/' + secure_filename(f.filename))
        faces = face_cascade.detectMultiScale(pic, scaleFactor=1.11, minNeighbors=8)


        for (x, y, w, h) in faces :
            img = pic[y : y + h, x : x + w]
            img = cv2.resize(img, (224, 224))
            

            predict_gender = model_gender.predict(np.array(img).reshape(-1, 224, 224, 3))
            gender = np.round(predict_gender)

            predict_race = model_race.predict(np.array(img).reshape(-1, 224, 224, 3))
            race = np.argmax(predict_race)

            
            predict_age = model_age.predict(np.array(img).reshape(-1, 224, 224, 3))
            predict_age = int(np.round(predict_age))
            
            if gender == 0 :
                gend = 'Female'
            else :
                gend = 'Male'
                
            if race == 0 :
                rac = 'Asian'
                
            elif race == 1 :
                rac = 'Black'
                
            else :
                rac = 'White'
            
            cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 225, 0), 1)
            cv2.putText(pic, str(gend) + '/' + str(predict_age) +'/' + str(rac), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, w * 0.005, (0, 0, 255), 1)

        pic1 = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20, 10))
        plt.axis('off')
        plt.imshow(pic1)
        plt.show()


        
        return redirect('/')






if __name__ == '__main__' :
    app.run(debug=True)











