import streamlit as st
import time
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("Caching Demo")

st.button("Test Caching")

st.subheader("st.cache_data")

@st.cache_data

def cache_this():
    time.sleep(2)
    out = "I'm done running"
    return out

out = cache_this()

st.write(out)

st.subheader("st.cache_resource")

@st.cache_resource
def create_lr():
    time.sleep(2)
    x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
    y = np.array([1, 2, 3, 4, 5, 6, 7])

    model = LinearRegression().fit(x, y)
    return model


lr = create_lr()

x_pred = np.array([8]).reshape(-1, 1)
pred = lr.predict(x_pred)

st.write(f"The prediction is: {pred[0]}")