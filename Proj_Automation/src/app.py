import streamlit as st
import requests
import os

st.title('Salary Prediction App')

# API endpoint
API_URL = os.getenv('API_URL', 'http://localhost:5000/predict')

def predict_salary(years_experience):
    """Call prediction API"""
    try:
        response = requests.post(
            API_URL,
            json={'years_experience': years_experience}
        )
        response.raise_for_status()
        return response.json()['predicted_salary']
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {e}")
        return None

# User input
years_exp = st.slider('Years of Experience', 0.0, 30.0, 5.0)

# Prediction
if st.button('Predict Salary'):
    with st.spinner('Predicting...'):
        salary = predict_salary(years_exp)
        if salary:
            st.success(f'Predicted Salary: ${salary:,.2f}')

# Model info
st.sidebar.title('About')
st.sidebar.info('This app predicts software engineer salaries based on years of experience using a trained ML model.')
st.sidebar.markdown('**MLOps Pipeline:**')
st.sidebar.markdown('- Data versioning with DVC')
st.sidebar.markdown('- Model training with scikit-learn')
st.sidebar.markdown('- API serving with Flask')
st.sidebar.markdown('- CI/CD with GitLab')