!pip install joblib
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st


import streamlit as st

# Load the model from file
model = joblib.load('C:/Users/kalaz/model.joblib')

# Define your Streamlit app
def app():
    st.title('My Streamlit App')
    # Use the loaded model in your app
    prediction = model.predict([[98039, 4, 5000]])
    st.write(f"Predicted price: {prediction[0]}")

# Run your Streamlit app
if __name__ == '__main__':
    app()
