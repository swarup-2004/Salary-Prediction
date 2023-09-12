import streamlit as st
import pickle
import numpy as np

def load_model():
    with open("saved_steps.pkl", 'rb') as file:
        data = pickle.load(file)
        
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]

def show_predict_page():
    st.title("Software Developoer Salary Prediction")

    st.write(("### We need to predict the salary"))

    countries = {
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden"
    }

    country = st.selectbox("Country",countries )
    experience = st.slider("Years of experience", 0, 50, 3)
    ok = st.button("Calculate Salary")

    if ok:
        x = np.array([[country, experience]])
        x[:,0] = le_country.transform(x[:,0])
        #x = x.astype(float)

        salary = regressor.predict(x)
        st.subheader(f"The Estimated Salary is ${salary[0]:.2f}")
