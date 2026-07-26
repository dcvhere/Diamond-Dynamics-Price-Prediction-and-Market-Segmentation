# Diamond Price Prediction & Market Segmentation App

## app deployed using streamlit

[Diamond Price Prediction & Market Segmentation App](https://diamond-dynamics-price-prediction-and-market-segmentation-01.streamlit.app/)

## Overview
This project is a Machine Learning-based web application that predicts diamond prices and segments diamonds into different market categories. It uses regression for price prediction and KMeans clustering for market segmentation, all wrapped inside an interactive web app built with Streamlit.

## Features
- Predict diamond price based on input features  
- Segment diamonds into market categories using clustering  
- Simple and interactive UI using Streamlit  
- Real-time predictions  

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib / Seaborn  

## Project Structure
Diamond Dynamics Project/
│
├── app.py                  # Streamlit application  
├── requirements.txt        # Dependencies  
├── README.md               # Documentation  
│── diamonds.csv            # Dataset
|── model.pkl               # Saved model

## Installation & Setup (Run Locally)

### 1. Download the project
Download or clone this repository to your local machine.

### 2. Install dependencies
Open terminal inside the project folder and run:
pip install -r requirements.txt

### 3. Run the Streamlit app
streamlit run app.py

### 4. Open in browser
The app will run at:
http://localhost:8501

## Dataset
The dataset contains diamond attributes such as:
- Carat  
- Cut  
- Color  
- Clarity  
- Depth  
- Table  
- Dimensions (x, y, z)  

After downloading, place it in:
data/diamonds.csv

## Models Used
- Regression Model → Predicts diamond price  
- KMeans Clustering → Segments diamonds into market groups  
 

## Note
This project is created for learning and demonstration purposes.
