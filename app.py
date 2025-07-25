from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Load the original data to get unique values for dropdowns
data = pd.read_csv("houseData.csv")
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
data = data[cols_to_use]

# Fill missing values as done in the notebook
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
data[cols_to_fill_zero] = data[cols_to_fill_zero].fillna(0)
data['Landsize'] = data['Landsize'].fillna(data.Landsize.mean())
data['BuildingArea'] = data['BuildingArea'].fillna(data.BuildingArea.mean())
data.dropna(inplace=True)

# Get unique values for dropdowns
suburbs = sorted(data['Suburb'].unique())
types = sorted(data['Type'].unique())
methods = sorted(data['Method'].unique())
seller_gs = sorted(data['SellerG'].unique())
regions = sorted(data['Regionname'].unique())
council_areas = sorted(data['CouncilArea'].unique())

@app.route('/')
def home():
    return render_template('index.html', 
                         suburbs=suburbs,
                         types=types,
                         methods=methods,
                         seller_gs=seller_gs,
                         regions=regions,
                         council_areas=council_areas)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Create a DataFrame with the form data
        input_data = pd.DataFrame([form_data])
        
        # Convert numerical fields to appropriate types
        numerical_fields = ['Rooms', 'Propertycount', 'Distance', 'Bedroom2', 
                          'Bathroom', 'Car', 'Landsize', 'BuildingArea']
        for field in numerical_fields:
            input_data[field] = pd.to_numeric(input_data[field])
        
        # One-hot encode categorical variables
        input_data = pd.get_dummies(input_data)
        
        # Ensure all columns from training are present
        # First get the model's expected features (this would need to be saved with the model)
        # For now, we'll use a simplified approach
        expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        
        # If we know the expected features, align the input data
        if expected_features is not None:
            # Add missing columns with 0
            for feature in expected_features:
                if feature not in input_data.columns:
                    input_data[feature] = 0
            
            # Ensure the order of columns matches
            input_data = input_data[expected_features]
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Format the prediction for display
        predicted_price = "${:,.2f}".format(prediction[0])
        
        return render_template('results.html', 
                             prediction=predicted_price,
                             input_data=form_data)
    
    except Exception as e:
        return render_template('results.html', 
                             prediction="Error in prediction",
                             error=str(e))

if __name__ == '__main__':
    app.run(debug=True)