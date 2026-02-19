from flask import Flask, render_template, request
import numpy as np
from joblib import load
import os
import warnings
warnings.filterwarnings('ignore')

# Import sklearn modules BEFORE loading any models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import sklearn.tree._tree

app = Flask(__name__)

# Lazy load model and scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
model = None
scaler = None

def load_models():
    global model, scaler
    if model is None:
        model = load(os.path.join(base_dir, "floods.save"))
        scaler = load(os.path.join(base_dir, "transform.save"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('index.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/result', methods=['POST'])
def result():
    load_models()
    
    temp = float(request.form['temp'])
    humidity = float(request.form['humidity'])
    cloud = float(request.form['cloud'])
    annual = float(request.form['annual'])
    janfeb = float(request.form['janfeb'])
    marmay = float(request.form['marmay'])
    junsep = float(request.form['junsep'])
    octdec = float(request.form['octdec'])
    avgjune = float(request.form['avgjune'])
    sub = float(request.form['sub'])

    data = np.array([[temp, humidity, cloud, annual, janfeb, marmay, junsep, octdec, avgjune, sub]])
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        return render_template('imageprediction.html', result="Flood Chance")
    else:
        return render_template('imageprediction.html', result="No Flood Chance")

if __name__ == '__main__':
    app.run(debug=True)
