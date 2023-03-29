import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import joblib
from PIL import Image
import requests

#Load the model
model = joblib.load("heart_disease.joblib")

#Features
features = ['Sex','BMI','Age','Smoking','Alcohol Driking','Sleep Time','Physical Health','Mental Health','Physical Activity','Difficulty Walking']
sex_dict = {'Female':0,'Male':1}
age_dict = {'18-24':1,'25-29':2,'30-34':3,'35-39':4,'40-44':5,'45-49':6,'50-54':7,'55-59':8,'60-64':9,'65-69':10,'70-74':11,'75-79':12,'80 or older':13}
smoking_dict ={'Yes':1, 'No':0}
alc_dict ={'Yes':1, 'No':0}
p_act_dict = {'Yes':1, 'No':0}
diff_walk_dict = {'Yes':1, 'No':0}

#Layout of the App
st.set_page_config(page_title="Heart Disease Predictor for Diabetic Patients",page_icon=':heart:')

#st.title("Heart Disease Predictor")
st.markdown("<h1 style='text-align: center; padding-bottom: 0px ;'>Heart Disease Predictor</h1>", 
            unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 16px;'>Only Applicable for Diabetic Patients</h1>", 
            unsafe_allow_html=True)

disclaimer = """**Disclaimer**: This app is not intended to be a substitute for professional medical advice, diagnosis, or treatment. The predictions and information provided by the app are for educational and informational purposes only. The predictions are based on a model and may not always be accurate. Users should consult with a qualified healthcare provider before making any decisions based on the app's predictions or information."""

st.markdown(f"<p style='font-size: 12px'>{disclaimer}</p>", unsafe_allow_html=True)

#Add image of heart
image_url = 'https://raw.githubusercontent.com/SatyamedhasP/Heart_Disease_Prediction/main/Heart_Image2.jpg'
image = Image.open(requests.get(image_url, stream=True).raw)
st.image(image, use_column_width=True)

st.subheader('Input Features')

#Font function
def custom_header(text):
    return f"<h3 style='font-size: 20px; padding-bottom: 0px ; font-weight: normal;'>{text}</h3>"

#Sex input
selectbox_label = "Sex:"
unique_key = selectbox_label.replace(" ", "_") + "_selectbox1"
st.markdown(custom_header(selectbox_label), unsafe_allow_html=True)
sex_input = st.selectbox('',('Female', 'Male'),key=unique_key)
sex = sex_dict[sex_input]

#Age input
selectbox_label = "Age Category:"
unique_key = selectbox_label.replace(" ", "_") + "_selectbox2"
st.markdown(custom_header(selectbox_label), unsafe_allow_html=True)
age_input = st.selectbox('',('18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80 or older'),key=unique_key)
age = age_dict[age_input]

#BMI input
slider_label = "BMI:"
unique_key = slider_label.replace(" ", "_") + "_slider1"
st.markdown(custom_header(slider_label), unsafe_allow_html=True)
bmi = st.slider('',min_value=0.0, max_value=50.00, step=0.1, value=20.00,key=unique_key)

#Smoking
selectbox_label = "Smoking:"
unique_key = selectbox_label.replace(" ", "_") + "_selectbox3"
st.markdown(custom_header(selectbox_label), unsafe_allow_html=True)
smoking_input = st.selectbox('Have you smoked at least 100 cigarettes in your entire life?',('Yes', 'No'),key=unique_key)
smoking = smoking_dict[smoking_input]


#Alcohol Drinking
selectbox_label = "Alcohol Drinking:"
unique_key = selectbox_label.replace(" ", "_") + "_selectbox4"
st.markdown(custom_header(selectbox_label), unsafe_allow_html=True)
alc_input = st.selectbox('Select yes if you are male and have more than 14 drinks per week or select yes if you are female and have more than 7 drinks per week',('Yes', 'No'),key=unique_key)
alcohol = alc_dict[alc_input]
# st.write('Select yes if you are male and have more than 14 drinks per week')
# st.write('Select yes if you are female and have more than 7 drinks per week ')

#Sleep Time
slider_label = "Sleep Time:"
unique_key = slider_label.replace(" ", "_") + "_slider2"
st.markdown(custom_header(slider_label), unsafe_allow_html=True)
sleep = st.slider('On average, how many hours of sleep do you get in a 24-hour period?',min_value=0.0, max_value=24.00, step=1.00, value=7.00,key=unique_key)

#Physical Health
slider_label = "Physical Health:"
unique_key = selectbox_label.replace(" ", "_") + "_slider3"
st.markdown(custom_header(slider_label), unsafe_allow_html=True)
p_health = st.slider('Includes physical illness and injury, for how many days during the past 30 days was your physical health not good?',min_value=0.0, max_value=30.00, step=1.00, value=5.00,key=unique_key)

#Mental Health
selectbox_label = "Mental Health:"
unique_key = selectbox_label.replace(" ", "_") + "_selectbox5"
st.markdown(custom_header(selectbox_label), unsafe_allow_html=True)
m_health = st.slider('For how many days during the past 30 days was your mental health not good?',min_value=0.0, max_value=30.00, step=1.00, value=5.00,key=unique_key)

#Physical Activity
selectbox_label = "Physical Activity:"
unique_key = selectbox_label.replace(" ", "_") + "_selectbox6"
st.markdown(custom_header(selectbox_label), unsafe_allow_html=True)
p_act_input = st.selectbox('Have you been active in the past 30 days?',('Yes', 'No'),key=unique_key)
p_act = p_act_dict[p_act_input]


#Difficulty Walking
selectbox_label = "Difficulty Walking:"
unique_key = selectbox_label.replace(" ", "_") + "_selectbox7"
st.markdown(custom_header(selectbox_label), unsafe_allow_html=True)
diff_walk_input = st.selectbox('Do you have serious difficulty walking or climbing stairs?',('Yes', 'No'),key=unique_key)
diff_walk = diff_walk_dict[diff_walk_input]

#Prediction
data = pd.DataFrame({'BMI':bmi,
                     'Physical_Health':p_health,
                     'Mental_Health':m_health,
                     'Sleep_Time':sleep,
                     'Smoking':smoking,
                     'Alcohol_Drinking':alcohol,
                     'Difficulty_Walking':diff_walk,
                     'Sex': sex,
                     'Physical_Activity':p_act,
                     'Age': age}, index=[0])

                 
#Button Function
if st.button('Predict'):
    prediction = model.predict(data)[0]
    if prediction ==0:
        result_nhr = 'No heart disease risk'
        image_url_nhr = 'https://raw.githubusercontent.com/SatyamedhasP/Heart_Disease_Prediction/main/healthy_heart.jpg'
        image_nhr = Image.open(requests.get(image_url_nhr, stream=True).raw)
        st.write('Model Prediction:', result_nhr)
        st.image(image_nhr, use_column_width=True)
    else:
        result_hr = 'Heart disease risk'
        image_url_hr = 'https://raw.githubusercontent.com/SatyamedhasP/Heart_Disease_Prediction/main/risk.jpg'
        image_hr = Image.open(requests.get(image_url_hr, stream=True).raw)
        st.write('Model Prediction:', result_hr)
        st.image(image_hr, use_column_width=True)

    
