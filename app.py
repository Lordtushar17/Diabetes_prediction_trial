from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_prediction_model.pkl')

# Set up the scaler by reloading the dataset
diabetes_dataset = pd.read_csv('D:\\PICT Techfiesta\\diabetes_prediction\\diabetes.csv')
X = diabetes_dataset.drop(columns='Outcome', axis=1)
scaler = StandardScaler()
scaler.fit(X)  # Fit the scaler on the data

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # HTML form for user input

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form (assuming input fields are named appropriately)
        input_data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        # Convert the input data to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

        # Standardize the data using the scaler
        standardized_data = scaler.transform(input_data_as_numpy_array)

        # Make a prediction
        prediction = model.predict(standardized_data)

        # Interpret the result
        result = "Congrats, you do not have diabetes" if prediction[0] == 0 else "You have diabetes"

        return render_template('result.html', result=result)

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
