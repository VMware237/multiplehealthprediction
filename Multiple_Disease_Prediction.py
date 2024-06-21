#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:10:28 2024

@author: vaishnavimhaske
"""

import numpy as np
import pickle
import streamlit as st 
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="👩‍⚕️")


# Load the saved models and scalers
diabetes_model = pickle.load(open('/Users/vaishnavimhaske/Documents/Multiplt_Disease_Prediction/saved models/Disease_Prediction_Diabetes.sav', 'rb'))
diabetes_scaler = pickle.load(open('/Users/vaishnavimhaske/Documents/Multiplt_Disease_Prediction/saved models/scaler.sav', 'rb'))

heart_model = pickle.load(open('/Users/vaishnavimhaske/Documents/Multiplt_Disease_Prediction/saved models/Heart_Disease_Prediction.sav', 'rb'))
heart_scaler = pickle.load(open('/Users/vaishnavimhaske/Documents/Multiplt_Disease_Prediction/saved models/scaler_heart.sav', 'rb'))

parkinsons_model = pickle.load(open('/Users/vaishnavimhaske/Documents/Multiplt_Disease_Prediction/saved models/Parkinsons_Prediction.sav', 'rb'))
parkinsons_scaler = pickle.load(open('/Users/vaishnavimhaske/Documents/Multiplt_Disease_Prediction/saved models/scaler_parkinsons.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction WebApp',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['clipboard2-pulse','heart-pulse','virus2'],
                           default_index=0)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Watch: Early Detection for Better Health')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure')
    with col1:
        SkinThickness = st.text_input('Skin Thickness')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    with col2:
        Age = st.text_input('Age')

    if st.button('Diabetes Test Result'):
        input_data = np.asarray([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        input_data = input_data.reshape(1, -1)
        std_data = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(std_data)
        
        if prediction[0] == 0:
            st.success("The person does not have diabetes.")
        else:
            st.error("The person has diabetes.")

# Heart Disease Prediction
if selected == 'Heart Disease Prediction':
    st.title('CardioCheck: Know Your Heart\'s Health Instantly')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (1 = Male, 0 = Female)')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col2:
        exang = st.text_input('Exercise Induced Angina')
    with col3:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col1:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col2:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col3:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    if st.button('Heart Disease Test Result'):
        input_data = np.asarray([age, sex, cp, thalach, exang, oldpeak, slope, ca, thal])
        input_data = input_data.reshape(1, -1)
        std_data = heart_scaler.transform(input_data)
        prediction = heart_model.predict(std_data)
        
        if prediction[0] == 0:
            st.success("The person does not have heart disease.")
        else:
            st.error("The person has heart disease.")

# Parkinson's Disease Prediction
if selected == 'Parkinsons Prediction':
# page title
    st.title("NeuroHealth: Predictive Analysis for Parkinson's")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP\:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP\:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP\:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP\:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP\:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP\:RAP')

    with col2:
        PPQ = st.text_input('MDVP\:PPQ')

    with col3:
        DDP = st.text_input('Jitter\:DDP')

    with col4:
        Shimmer = st.text_input('MDVP\:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP\:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer\:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer\:APQ5')

    with col3:
        APQ = st.text_input('MDVP\:APQ')

    with col4:
        DDA = st.text_input('Shimmer\:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

        
    if st.button('Parkinson\'s Disease Test Result'):
        input_data = np.asarray([fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])
        input_data = input_data.reshape(1, -1)
        std_data = parkinsons_scaler.transform(input_data)
        prediction = parkinsons_model.predict(std_data)
        
        if prediction[0] == 0:
            st.success("The person does not have Parkinson's disease.")
        else:
            st.error("The person has Parkinson's disease.")
