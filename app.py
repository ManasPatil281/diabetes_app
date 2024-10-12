import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

st.markdown("""
    <style>
    body {
        background-color: #f0f0f5;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
    }
    h1, h2, h3 {
        color: #2a9d8f;
        font-family: 'Arial', sans-serif;
    }
    .stButton button {
        background-color: #2a9d8f;
        color: white;
        font-size: 18px;
        border-radius: 8px;
    }
    .stTextInput input {
        border-radius: 10px;
        border: 2px solid #2a9d8f;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Diabetes Prediction App")
st.markdown("## Powered by Machine Learning  ")

diabetes_dataset = pd.read_csv('diabetes.csv')

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

st.markdown("### Enter the following details:")

input_data = {
    'Pregnancies': st.number_input('Pregnancies', min_value=0, max_value=20, value=1),
    'Glucose': st.number_input('Glucose', min_value=0, max_value=300, value=120),
    'BloodPressure': st.number_input('Blood Pressure', min_value=0, max_value=200, value=70),
    'SkinThickness': st.number_input('Skin Thickness', min_value=0, max_value=100, value=20),
    'Insulin': st.number_input('Insulin', min_value=0, max_value=900, value=80),
    'BMI': st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0),
    'DiabetesPedigreeFunction': st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5),
    'Age': st.number_input('Age', min_value=0, max_value=120, value=30)
}

input_data_as_array = np.asarray(list(input_data.values())).reshape(1, -1)
standardized_input_data = scaler.transform(input_data_as_array)

if st.button('Predict Diabetes Status'):
    prediction = classifier.predict(standardized_input_data)

    if prediction[0] == 0:
        st.success('🟢 The person is **not diabetic**.')
    else:
        st.error('🔴 The person **is diabetic**.')

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

st.markdown(f"### Accuracy score on test data: **{test_data_accuracy:.2f}**")
