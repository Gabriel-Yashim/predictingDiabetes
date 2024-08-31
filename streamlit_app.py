# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 00:09:17 2023

@author: Gabriel Yashim
"""

import numpy as np 
import pickle
import streamlit as st


model = pickle.load(open('XGBoost_model.pkl', 'rb'))


st.title('Diabetes Prediction System')
html_temp = """
    <h3 style="color:white;text-align:center;"></h3>
    <div style="background-color:rgb(117, 3, 3);padding:10px;margin-bottom:3rem">
        <p style="text-align:justify;">
            Welcome to this simple Diabetes Prediction System. The system can tell if a person has diabetes or not based on the following features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Diabetes Pedigree Function,	and Age <br> 
            Fill the fields below with the right data to get predictions.
        </p>  
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)



# Input field for column1
age = st.text_input("Enter Your Age (an integer)")
prg = st.text_input("Number of Pregnancies (an integer)")
glu = st.text_input("Enter Your Glucose Level (an integer)")
bp = st.text_input("Enter Your Blood Pressure Level (an integer)")
skin = st.text_input("Enter Your Skin Thickness (an integer)")
ins = st.text_input("Enter Your Insulin Level (an integer)")
bmi = st.text_input("Enter Your BMI(a float)")
dpf = st.text_input("Enter Your Diabetes Pedigree Function (a float)")
    


diabetes_pred = ''

output = ''


if st.button('Submit'):
    input_data = np.array([[prg,glu,bp,skin,ins,bmi,dpf,age]], dtype=np.float32)
    diabetes_pred = model.predict(input_data)
    
    if diabetes_pred[0] == 1:
        output = 'Diabetic'
        st.write(f"Result: Ops, it seems you are {output}.")
    else:
        output = 'Not Diabetic'
        st.write(f"Result: You are {output}.")
        
    
    




