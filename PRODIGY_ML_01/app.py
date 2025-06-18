from flask import Flask, request, render_template
import joblib
import numpy as np


app = Flask(__name__)

# Load the saved model (replace with your model path)
model = joblib.load('Property_Price_Predictor.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Display the form to input data

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the user inputs from the form
            Total_sqft = float(request.form['size'])
            bedrooms = float(request.form['bedrooms'])
            bath = float(request.form['bathrooms'])

            # Prepare the features for prediction (make sure to match the model's expected input format)
            features = np.array([Total_sqft, bedrooms, bath]).reshape(1, -1)

            # Make the prediction
            prediction = model.predict(features)

            # Display the result
            return render_template('index.html', prediction_text=f'Predicted Price: {prediction[0]:,.2f} Lakhs')
        except Exception as e:
            return render_template('index.html', prediction_text="Error in prediction. Please check the input.")

if __name__ == '__main__':
    app.run(debug=True)
