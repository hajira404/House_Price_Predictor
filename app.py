from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model, scaler, and R² score from the pickle file
with open('model_and_scaler.pkl', 'rb') as file:
    model, scaler, r2 = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        gr_liv_area = float(request.form['GrLivArea'])
        bedroom_abv_gr = int(request.form['BedroomAbvGr'])
        total_bathrooms = float(request.form['TotalBathrooms'])  # Assuming it can be float

        # Create a DataFrame for input data
        input_data = pd.DataFrame({
            'GrLivArea': [gr_liv_area],
            'BedroomAbvGr': [bedroom_abv_gr],
            'TotalBathrooms': [total_bathrooms]
        })

        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Make predictions
        predicted_price = model.predict(scaled_data)

        # Prepare the prediction text
        prediction_text = f'Predicted Price for the house: ${predicted_price[0]:,.2f}'

        # Prepare the accuracy text
        accuracy_text = f'Model R² Score: {r2:.2f}'

        return render_template(
            'index.html',
            prediction_text=prediction_text,
            accuracy_text=accuracy_text
        )
    except Exception as e:
        # Handle errors gracefully
        error_text = f'Error: {str(e)}'
        return render_template('index.html', prediction_text=error_text)

if __name__ == '__main__':
    app.run(debug=True)