import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    
    data_array = np.array(list(data.values())).reshape(1, -1)
    
    output = model.predict(data_array)
    # print(output[0]) 
    
    return jsonify(output[0])



@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x)for x in request.form.values()]
    final_input = data = np.array(data).reshape(1,-1)

    output = model.predict(final_input)[0]
    
    return render_template("home.html",prediction_text = "The house prediction price is ₹ {}".format(output))


if __name__ == "__main__":
    app.run()