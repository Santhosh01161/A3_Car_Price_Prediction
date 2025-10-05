import numpy as np
from flask import Flask, request, render_template
import cloudpickle
import pickle
import os

# Create the Flask application
app = Flask(__name__)

# Mock classes for testing when models can't be loaded
class MockModel:
    def predict(self, X):
        return ["Test Class"]

class MockScaler:
    def transform(self, X):
        return X

# Load your pre-trained machine learning model and scaler
model = None
scaler = None

if os.path.exists("cppm_a3_model.pkl"):
    try:
        with open("cppm_a3_model.pkl", "rb") as f:
            model = cloudpickle.load(f)
    except Exception as e:
        print(f"Cloudpickle failed, trying regular pickle: {e}")
        try:
            with open("cppm_a3_model.pkl", "rb") as f:
                model = pickle.load(f)
        except Exception as e2:
            print(f"Both pickle methods failed: {e2}")
            model = MockModel()
else:
    model = MockModel()

if os.path.exists("cppm_a3_scaler.pkl"):
    try:
        with open("cppm_a3_scaler.pkl", "rb") as f:
            scaler = cloudpickle.load(f)
    except Exception as e:
        print(f"Cloudpickle failed, trying regular pickle: {e}")
        try:
            with open("cppm_a3_scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
        except Exception as e2:
            print(f"Both pickle methods failed: {e2}")
            scaler = MockScaler()
else:
    scaler = MockScaler()


@app.route('/', methods=['GET'])
def home():
    """Renders the main page (index.html) when a user first visits."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives user input from the form, makes a prediction,
    and renders the result back to the page.
    """
    try:
        # Get the values from the form fields and convert them to floats
        max_power = float(request.form['maxpower'])
        km_driven = float(request.form['kmdriven'])
        year = float(request.form['Year'])

        # Prepare features for the model
        features = np.array([[max_power, km_driven, year]])
        features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)

        # Format prediction
        prediction_text = f"Predicted Car Class is {prediction[0]}"

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        # Handle any errors gracefully
        error_message = f"Error: {str(e)}"
        return render_template('index.html', prediction_text=error_message)


if __name__ == '__main__':
    # Run the Flask app in debug mode for development
    app.run(host='0.0.0.0', port=8080, debug=True)
