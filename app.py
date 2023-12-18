from flask import Flask, render_template, request, jsonify
import os 
import numpy as np
import pandas as pd
from src.mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/')  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


# Handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the request
        data = request.get_json()


        # Define feature names based on your dataset
        feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                         'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
                         'yr_built', 'city', 'statezip']

        # Validate input data
        if not all(feature in data for feature in feature_names):
            raise ValueError("Missing or incorrect features in input data.")

        # Process input data and perform prediction using your model
        # Ensure that the order of features matches the order used during training
        input_data = pd.DataFrame({feature: [data[feature]] for feature in feature_names})
            
        obj = PredictionPipeline()
        predicted_price = obj.predict(input_data)

        # Return the predicted price as JSON
        return jsonify({"predicted_price": predicted_price[0]})

    except Exception as e:
        # Log detailed error information
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


 


if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)
     # app.run(host="0.0.0.0", port = 8080, debug=True)