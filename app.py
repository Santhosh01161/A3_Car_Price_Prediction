
import numpy as np
from flask import Flask, request, render_template
import pickle

# Create the Flask application
app = Flask(__name__)

# Load your pre-trained machine learning model from the .pkl file
# Ensure the path "model/car-price-prediction.pkl" is correct
model = pickle.load(open("cppm_a3_model.pkl", "rb"))
scaler = pickle.load(open("cppm_a3_scaler.pkl", "rb"))

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

        # Create a NumPy array with the features in the correct order for your model
        features = np.array([[max_power, km_driven, year]])
        features = scaler.transform(features)
        
        # Use the loaded model to make a prediction
        prediction = model.predict(features)
    


        # Format the prediction to be more user-friendly (e.g., two decimal places)

        prediction_text = f"Predicted Car Class is {prediction[0]}"

        # Render the index.html page again, passing the prediction text to it
        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        # Handle any errors during conversion or prediction
        error_message = f"Error: {str(e)}"
        return render_template('index.html', prediction_text=error_message)

if __name__ == '__main__':
    # Run the Flask app in debug mode for development
    app.run(host='0.0.0.0',port=8080,debug=True)
