import numpy as np
from flask import Flask, request, render_template
import cloudpickle

# Create the Flask application
app = Flask(__name__)

# Load your pre-trained machine learning model and scaler using cloudpickle
# This avoids version mismatch issues with pickle (e.g., numpy, sklearn upgrades)
with open("cppm_a3_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

with open("cppm_a3_scaler.pkl", "rb") as f:
    scaler = cloudpickle.load(f)


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
