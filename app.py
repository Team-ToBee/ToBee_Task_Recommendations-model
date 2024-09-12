from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model
model_path = os.environ.get('MODEL_PATH', 'productivity_model.joblib')
model = joblib.load(model_path)

# Define category and priority encoders
categories = ['Work', 'Personal', 'Study', 'Health', 'Social']
cat_encoder = {cat: code for code, cat in enumerate(categories)}
pri_encoder = {'Low': 0, 'Medium': 1, 'High': 2}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Preprocess the input data
    category_encoded = cat_encoder[data['category']]
    priority_encoded = pri_encoder[data['priority']]
    days_until_due = (pd.to_datetime(data['due_date']) - pd.to_datetime(data['creation_date'])).days
    day_of_week = pd.to_datetime(data['creation_date']).dayofweek
    hour_of_day = pd.to_datetime(data['creation_date']).hour
    
    features = np.array([[category_encoded, priority_encoded, data['estimated_duration'], 
                          days_until_due, day_of_week, hour_of_day]])
    
    # Make prediction
    probability = model.predict_proba(features)[0][1]  # Probability of completion
    recommended_priority = 'High' if probability < 0.5 else 'Normal'
    
    # Prepare response
    response = {
        'task': data['description'],
        'completion_probability': float(probability),
        'recommended_priority': recommended_priority
    }
    
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)