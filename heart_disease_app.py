# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 09:32:12 2025

@author: HP
"""

import numpy as np
import pickle
import streamlit as st


heart_disease_1 = pickle.load(open("C:/Users/HP/OneDrive/Desktop/Deploying machine learning/heart_disease1.sav","rb"))
                                

def heart_prediction(input_data):
    input_data_as_np_array = np.asarray(input_data).reshape(1,-1)
    prediction = heart_disease_1.predict(input_data_as_np_array)
    prediction
    
    if prediction[0] == 0:
        return "The preson has not heart disease"
    else:
        return "The person has heart disease"
        
def main():
    st.title("Heart Disease Prediction Web App")
    
    age = st.number_input("Age")
    sex = st.number_input("Sex (1 = Male, 0=Female)")
    cp = st.number_input("Chest Pain type(cp)")
    trestbps = st.number_input("Resting Blood Presure(trestbps)")
    chol = st.number_input("Serum Cholestrol (mg/dl)")
    fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (1=Yes, 0 = No)(fbs)")
    restecg = st.number_input("Resting ECG results(0-2)(restecg)")
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)")
    exang = st.number_input("Exercise Included Angina (1= Yes, 0=No)(exang)")
    oldpeak = st.number_input("ST depression induced by excerise (oldpeak)")
    slope =  st.number_input("Slope of ST segment (0-2)(slope)")
    ca = st.number_input("Number of major vessels (0-3)(ca)")
    thal = st.number_input("Thalassemia(thal)")
    
 
    heart = ""
    if st.button("predict"):
        heart = heart_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
                            
    st.success(heart)

if __name__ == "__main__":
    main()


