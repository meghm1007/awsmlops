from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    gpa = float(request.form.get('gpa'))
    iq = int(request.form.get('iq'))
    skill = int(request.form.get('skill'))
    
    result = model.predict(np.array([gpa, iq, skill]).reshape(1, 3))
    
    if result[0]==1:
        result = 'Placed'
    else:
        result = 'Not Placed'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)