import streamlit as st
import pickle
import numpy as np

# MUST be first Streamlit command
st.set_page_config(page_title="Addiction_Prediction", page_icon=":shark:", layout="wide")

st.title("Addiction Prediction App")

def load_model():
    return pickle.load(open("linear_regression_model.pkl", "rb"))

model = load_model()

value1 = st.number_input("Hours spent on social media")

value2 = st.selectbox("Affects Academic Performance", [0, 1])

value3 = st.number_input("Sleep Hours Per Night")

value4 = st.selectbox("Mental Health Score", list(range(11)))

value5 = st.selectbox("Conflicts Over Social Media",list(range(11)))

if st.button("Predict Addiction Score"):

    input_data = np.array([[value1, value2, value3, value4, value5]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Addiction Score: {prediction[0]:.2f}")
