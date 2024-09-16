from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator

app = Flask(__name__)

# Load the trained model
model = joblib.load('productivity_model2.joblib')

# Define category and priority encoders
categories = ['Work', 'Personal', 'Study', 'Health', 'Social']
cat_encoder = {cat: code for code, cat in enumerate(categories)}
pri_encoder = {'Low': 0, 'Medium': 1, 'High': 2}

def safe_predict(model, features):
    """Safely predict using the model, handling various model types."""
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(features)[0][1]
        except IndexError:
            # If predict_proba returns a 1D array (e.g., for binary classification)
            return model.predict_proba(features)[0]
    elif hasattr(model, 'predict'):
        prediction = model.predict(features)[0]
        # If the prediction is binary (0 or 1)
        if prediction in [0, 1]:
            return float(prediction)
        # If the prediction is a probability
        elif 0 <= prediction <= 1:
            return prediction
        else:
            # If it's a multi-class prediction, we'll treat it as binary
            # by checking if it's the positive class (assumed to be 1)
            return float(prediction == 1)
    else:
        raise ValueError("Model doesn't have predict or predict_proba method")

def get_recommendations(user_tasks, model, cat_encoder, pri_encoder):
    recommendations = []
    for task in user_tasks:
        category_encoded = cat_encoder.get(task['category'], -1)  # -1 for unknown categories
        priority_encoded = pri_encoder.get(task['priority'], -1)  # -1 for unknown priorities
        days_until_due = (pd.to_datetime(task['due_date']) - pd.to_datetime(task['creation_date'])).days
        day_of_week = pd.to_datetime(task['creation_date']).dayofweek
        hour_of_day = pd.to_datetime(task['creation_date']).hour

        features = np.array([[category_encoded, priority_encoded, task['estimated_duration'],
                              days_until_due, day_of_week, hour_of_day]])
        
        try:
            probability = safe_predict(model, features)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            probability = 0.5  # default to 0.5 if prediction fails

        recommendations.append({
            'task': task['description'],
            'completion_probability': float(probability),
            'recommended_priority': 'High' if probability < 0.5 else 'Normal'
        })

    return sorted(recommendations, key=lambda x: x['completion_probability'])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'tasks' not in data:
        return jsonify({'error': 'No tasks provided'}), 400

    try:
        recommendations = get_recommendations(data['tasks'], model, cat_encoder, pri_encoder)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_type': type(model).__name__}), 200

if __name__ == '__main__':
    app.run(debug=True)