from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
model = joblib.load('model.joblib')

new_data = [[98102, 3, 1500]]
prediction = model.predict(new_data)
st.write(prediction)
