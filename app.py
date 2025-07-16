from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'saved_models/model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_columns = ['AGE','APOE4', 'CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'RAVLT_immediate_bl',
                 'RAVLT_learning_bl', 'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'FAQ_bl',
                 'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl',
                 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', ]
    
    # Extract data from form
    int_features = [float(x) for x in request.form.values()]
    input_df = pd.DataFrame([int_features], columns=input_columns)
    
    # Make prediction
    prediction = model.predict(input_df)
    output = prediction[0]

    return render_template('index.html', prediction_text='Predicted Score: {:.2f}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)