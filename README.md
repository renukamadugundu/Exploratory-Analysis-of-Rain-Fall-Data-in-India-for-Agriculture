# Exploratory-Analysis-of-Rain-Fall-Data-in-India-for-Agriculture
The Exploratory Analysis of Rainfall Data in India for Agriculture is a comprehensive study aimed at analyzing historical rainfall data across different regions in India. This project utilizes data visualization techniques, statistical analysis, and machine learning algorithms to gain insights into rainfall patterns.
1️⃣ Data Collection
✔ Option 1: Collect Dataset

IMD Rainfall Dataset

Kaggle rainfall dataset

Government Open Data Portal

✔ Option 2: Create Dataset

Manually create rainfall records (CSV format)

Include features like:

Year

State

Monthly rainfall

Annual rainfall

Rainfall category (Drought/Normal/Flood)

2️⃣ Data Pre-processing
📌 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
📌 Step 2: Import Dataset
df = pd.read_csv("rainfall_data.csv")
print(df.head())
📌 Step 3: Check for Null Values
print(df.isnull().sum())
📌 Step 4: Data Visualization
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.show()
📌 Step 5: Handling Missing Data
df.fillna(df.mean(), inplace=True)

OR

df.dropna(inplace=True)
📌 Step 6: Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
📌 Step 7: Splitting Data into Train & Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
3️⃣ Model Building
📌 Step 1: Import ML Libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
📌 Step 2: Initialize Model
For Regression:
model = LinearRegression()
For Classification:
model = RandomForestClassifier()
📌 Step 3: Train the Model
model.fit(X_train, y_train)
📌 Step 4: Test the Model
y_pred = model.predict(X_test)
4️⃣ Model Evaluation
✔ For Regression
from sklearn.metrics import r2_score, mean_squared_error

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
✔ For Classification
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
5️⃣ Save the Model
import pickle

pickle.dump(model, open("rainfall_model.pkl", "wb"))
6️⃣ Application Building (Flask Deployment)
📁 Project Folder Structure
Rainfall_Project/
│
├── model/
│   └── rainfall_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│
├── app.py
└── requirements.txt

📄 Step 1: Create HTML File (templates/index.html)
<!DOCTYPE html>
<html>
<head>
    <title>Rainfall Prediction</title>
</head>
<body>
    <h2>Rainfall Prediction System</h2>
    <form action="/predict" method="post">
        <input type="text" name="feature1" placeholder="Enter value">
        <input type="submit" value="Predict">
    </form>
    <h3>{{ prediction_text }}</h3>
</body>
</html>
🐍 Step 2: Build Flask Python Code (app.py)
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/rainfall_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    final_input = np.array([input_features])
    
    prediction = model.predict(final_input)
    
    return render_template("index.html",
                           prediction_text="Prediction: {}".format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
🚀 Final Output Flow
User Input → Flask App → Load Model → Predict → Display Result
🎓 Final Deliverables

✔ Dataset
✔ Jupyter Notebook (EDA + ML)
✔ Saved Model (.pkl)
✔ Flask Web Application
✔ Project Report
✔ GitHub Repository


🌧️ Rainfall Prediction – Complete Project Structure

Below is the recommended folder structure for your Rainfall Prediction project (IBM Deployment + Local Flask Application).

📁 Overall Project Structure
Rainfall_Prediction_Project/
│
├── IBM_Endpoint_Deploy/
│   │
│   ├── templates/
│   │   ├── index.html
│   │   ├── chance.html
│   │   └── noChance.html
│   │
│   ├── app.py
│   ├── Rainfall.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── imputer.pkl
│   └── requirements.txt
│
├── Rainfall_Prediction_Local/
│   │
│   ├── templates/
│   │   ├── index.html
│   │   ├── chance.html
│   │   └── noChance.html
│   │
│   ├── app.py
│   ├── Rainfall.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── imputer.pkl
│   └── requirements.txt
│
├── Rainfall_prediction.ipynb
└── Dataset/
    └── rainfall_data.csv
📌 Explanation of Each Component
1️⃣ IBM_Endpoint_Deploy Folder

Used for IBM Watson Cloud Deployment

Contains:

📂 templates/

index.html → User input page

chance.html → If rainfall chance is high

noChance.html → If rainfall chance is low

📄 app.py

Flask backend file

Loads model (.pkl)

Connects UI to ML model

📦 Model Files

Rainfall.pkl → Trained ML model

scaler.pkl → Feature scaling object

encoder.pkl → Categorical encoding object

imputer.pkl → Missing value handling object

📄 requirements.txt

Contains required libraries:

flask
numpy
pandas
scikit-learn
gunicorn
2️⃣ Rainfall_Prediction_Local Folder

Used for Running Flask App in Local System

Same structure as IBM folder but used locally.

Run using:

python app.py
3️⃣ Rainfall_prediction.ipynb

This is the Model Training Notebook

Contains:

Data Loading

Data Cleaning

Feature Engineering

Model Training

Model Evaluation

Saving Model

Example saving code inside notebook:

import pickle

pickle.dump(model, open("Rainfall.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))
pickle.dump(imputer, open("imputer.pkl", "wb"))
4️⃣ Model File Details
File Name	Purpose
Rainfall.pkl	Final trained ML model
scaler.pkl	StandardScaler / MinMaxScaler
encoder.pkl	LabelEncoder / OneHotEncoder
imputer.pkl	SimpleImputer for missing values
🌐 Application Workflow
🔄 Flow:

User → index.html → Flask app.py →
Imputer → Encoder → Scaler → Model →
Prediction → chance.html / noChance.html

🧠 How Prediction Logic Works in app.py

Simplified logic:

if prediction == 1:
    return render_template("chance.html")
else:
    return render_template("noChance.html")
🎯 Final Deliverables for Submission

✔ Rainfall_prediction.ipynb
✔ Dataset
✔ Rainfall.pkl
✔ scaler.pkl
✔ encoder.pkl
✔ imputer.pkl
✔ Flask Application Folder
✔ IBM Endpoint Deploy Folder
✔ Project Report
✔ PPT Presentation


Team Details
Team ID	LTVIP2026TMIDS71000
Team Leader	Kakerla Vishnu Priya
Team Member	 Renuka Madugundu
Team Member  Geethanjali Ediga
Team Member	Golla Manasa


📁 Overall Project Structure
Rainfall_Prediction_Project/
│
├── IBM_Endpoint_Deploy/
│   │
│   ├── templates/
│   │   ├── index.html
│   │   ├── chance.html
│   │   └── noChance.html
│   │
│   ├── app.py
│   ├── Rainfall.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── imputer.pkl
│   └── requirements.txt
│
├── Rainfall_Prediction_Local/
│   │
│   ├── templates/
│   │   ├── index.html
│   │   ├── chance.html
│   │   └── noChance.html
│   │
│   ├── app.py
│   ├── Rainfall.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── imputer.pkl
│   └── requirements.txt
│
├── Rainfall_prediction.ipynb
└── Dataset/
    └── rainfall_data.csv


    Technologies Used
Category	        Technology
Language:	         Python
ML Libraries:	     NumPy, Pandas, Scikit-learn
Visualization:	     Matplotlib, Seaborn
Model:	             Random Forest Regressor
Web Framework:	      Flask
API:	            OpenWeatherMap API
Frontend:	            HTML, CSS
Model Serialization:	  Joblib
Environment:	           Jupyter Notebook, VS Code

