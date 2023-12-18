# House Price Prediction MLflow

![Screenshot 2023-11-27 165610](https://github.com/Pramod858/House-Price-Prediction-MLflow/assets/80105491/822ca2c0-eebf-4ea4-9cd3-a75018b34170)

## Overview

This project encompasses a user-friendly web application designed for predicting house prices through a machine learning model trained on a diverse dataset. Leveraging MLflow, I've incorporated robust experiment tracking, allowing for efficient model versioning and reproducibility. Continuous Integration (CI) and Continuous Deployment (CD) pipelines have been implemented to automate testing and ensure seamless deployment. The application has been containerized using Docker, offering portability and scalability, and the Docker image has been seamlessly pushed to AWS Elastic Container Registry (ECR) for reliable deployment in a cloud environment.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python (version 3.9.18)
- Docker

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Pramod858/House-Price-Prediction-MLflow.git
   cd House-Price-Prediction-MLflow
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model and Run the Flask App Locally:**
   
    ```bash
   python main.py
   ```

   ```bash
   python app.py
   ```

   The application will be accessible at `http://127.0.0.1:8080/` in your browser.

## Using Docker

Alternatively, you can run the application using Docker.

1. **Build the Docker Image:**

   ```bash
   docker build -t house_price_prediction -f Dockerfile .
   ```

2. **Run the Docker Container:**

   ```bash
   docker run -p 8080:8080 house_price_prediction
   ```

   The application will be available at `http://127.0.0.1:8080/` in your browser.

## MLflow

https://dagshub.com/Pramod858/House-Price-Prediction-MLflow.mlflow


## Usage

- Open your browser and navigate to `http://127.0.0.1:8080/`.
- Input the details for the house (bedrooms, bathrooms, etc.).
- Click the "Predict Price" button to get the predicted house price.
