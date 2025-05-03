import streamlit as st
import joblib
import pandas as pd
import re

# Load model and vectorizer
model = joblib.load('models/email_classifier_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Preprocessing function (from notebook)
def preprocess_email(email_text):
    email_text = email_text.lower()
    email_text = re.sub(r'[^a-zA-Z\s]', '', email_text)
    email_text = re.sub(r'\s+', ' ', email_text).strip()
    return vectorizer.transform([email_text])

# Prediction function
def predict_email_category(email_text):
    vec = preprocess_email(email_text)
    pred = model.predict(vec)[0]
    return ['Ham', 'Spam', 'Phishing'][pred]

# Streamlit interface
st.title('Email Classifier: Ham, Spam, Phishing')
st.write('Enter an email message to classify it as Ham, Spam, or Phishing.')

email_input = st.text_area('Email Text', 'Enter email here...')
if st.button('Classify'):
    if email_input:
        prediction = predict_email_category(email_input)
        st.success(f'Prediction: **{prediction}**')
    else:
        st.error('Please enter an email.')
