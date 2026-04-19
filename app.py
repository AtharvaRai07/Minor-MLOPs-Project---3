import os
import sys
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

from config.paths_config import *

model = joblib.load(MODEL_FILE_PATH)
encoder = joblib.load(LABEL_ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)[0]

        predicted_class = encoder.inverse_transform([prediction])[0]

        return render_template('index.html', prediction=f"The predicted class is: {predicted_class}")

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
