import joblib
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import urllib.request


# Load the model from file
file_url = "https://drive.google.com/file/d/1hzsBNqhhZmSHBwGfBD86kMU2A9H7dFFV/view?usp=sharing"
file_id = file_url.split('/')[-2]
dwn_url = "https://drive.google.com/uc?id=" + file_id
urllib.request.urlretrieve(dwn_url, "model.joblib")
model = joblib.load("model.joblib")


# Define your Streamlit app
def app():
    st.title('My Streamlit App')
    # Use the loaded model in your app
    prediction = model.predict([[98039, 4, 5000]])
    st.write(f"Predicted price: {prediction[0]}")

# Run your Streamlit app
if __name__ == '__main__':
    app()
