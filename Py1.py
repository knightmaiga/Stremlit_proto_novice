import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pickle
import urllib.request
url ='https://github.com/knightmaiga/Stremlit_proto_novice/raw/main/Salary_Data.csv'
response = urllib.request.urlopen(url)

data=pd.read_csv(response)

x=data['YearsExperience']

st.write("X--> Years of Experience",x)

y=data['Salary']

st.write("Y--> Salary",y)

m=st.sidebar.radio("Menu",['Home','Prediction'])  

st.title("Salary Prediction")


if m=='Home':

    st.subheader("EDA")

    st.write("Shape")

    st.write(data.shape)

    st.write("Head")

    st.write(data.head())

    st.write("Dataset Information")

    st.write(data.describe())

    fig, ax=plt.subplots(figsize=(10,5))

    plt.xlabel("Years of Experience")

    plt.ylabel("Salary")

    plt.title("Years of Experience VS Salary")

    plt.scatter(x,y)

    st.pyplot(fig)

elif m=='Prediction':

    st.subheader("PREDICTION")

    from sklearn.model_selection import train_test_split

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

    x=np.array(x).reshape(-1,1)

    y=np.array(y).reshape(-1,1)

    from sklearn.linear_model import LinearRegression

    regressor=LinearRegression()

    regressor.fit(x,y)

    exp=st.number_input("Experience in Years:",0,42,1)

    exp=np.array(exp).reshape(1,-1)

    prediction=regressor.predict(exp)[0]

    if st.button("Salary Prediction"):

        st.write(f"{prediction}")