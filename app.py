import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

# Streamlit web app
st.title('Heart Disease Prediction')

# Input fields for patient details
age = st.number_input('Age', min_value=0, max_value=120, value=25)
sex = st.selectbox('Sex', options=['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)
chol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
restecg = st.selectbox('Resting ECG', options=[0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved', min_value=0, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Flourosopy', options=[0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])

# Convert categorical inputs to numerical values
sex = 1 if sex == 'Male' else 0
cp = int(cp)
fbs = int(fbs)
restecg = int(restecg)
exang = int(exang)
slope = int(slope)
ca = int(ca)
thal = int(thal)

# Prepare input data
input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)

# Display the prediction
if st.button('Predict'):
    if prediction[0] == 1:
        st.write('The patient is likely to have heart disease.')
    else:
        st.write('The patient is unlikely to have heart disease.')
