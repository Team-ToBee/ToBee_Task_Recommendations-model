# ToBee_Task_Recommendations-model
## Productivity Task Predictor

This project implements a machine learning model to predict the likelihood of completing tasks based on various features. It includes a trained Random Forest Classifier model and a Flask API to serve predictions.

## Features

- Predicts task completion probability
- Recommends task priority based on the prediction
- Flask API for easy integration with other applications

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Team-ToBee/ToBee_Task_Recommendations-model
   cd productivity-task-predictor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Flask API

1. Ensure you have the trained model file `productivity_model.joblib` in the project directory.

2. Start the Flask server:
   ```
   python app.py
   ```

3. The API will be available at `http://localhost:5000`.

### Making Predictions

Send a POST request to `http://localhost:5000/predict` with a JSON payload containing task information:

```json
{
  "category": "Work",
  "description": "Complete project report",
  "creation_date": "2023-09-15 09:00:00",
  "due_date": "2023-09-20 17:00:00",
  "priority": "High",
  "estimated_duration": 180
}
```

The API will respond with a prediction:

```json
{
  "task": "Complete project report",
  "completion_probability": 0.75,
  "recommended_priority": "Normal"
}
```

## Model Training

The model was trained on synthetic data generated to simulate task completion patterns. It uses the following features:

- Task category
- Task priority
- Estimated duration
- Days until due
- Day of week
- Hour of day

For details on the model training process, refer to the original script in the repository.

## Deployment

This project is configured for easy deployment on Railway. Follow these steps to deploy:

1. Fork this repository to your GitHub account.
2. Create a new project on Railway and connect it to your GitHub repository.
3. Railway will automatically detect the Python environment and install dependencies.
4. Set up the necessary environment variables in Railway's dashboard.
5. Deploy the application.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
