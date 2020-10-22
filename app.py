import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

st.write("""
# Simple Marks Prediction App
This app predicts the **Marks** scored by the student!
""")

st.sidebar.markdown('''
            By Kuldeep Sharma aka [SoleCodr](https://github.com/SoleCodr) \n
            GitHub Repo for the [App](https://github.com/SoleCodr/marks-prediction-app)
    ''')
st.write('''
        **Assuming the study hours to be maximum 10 hours.**
        ''')
    

def user_input():
    hr = st.number_input("Number of Hours of Study",0.00,10.00,6.2)
    hr = np.array([[hr]]).astype(np.float64)
    return hr

pred = user_input()

#Loading dataset
df = pd.read_csv("Data.txt")
#We create a prediction regression model for a better accuracy
model = Sequential()
model.add(Dense(2,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1)) #Final prediction layer
model.compile(loss='mse',optimizer='adam') #Compile the model
x = df.iloc[:,:-1].values
y = df.iloc[:,1].values
model.fit(x,y,epochs=300,batch_size=8) #Train the model for 300 epochs
prediction = round(float(model.predict(pred)))

if pred > 9.97:
    st.button("Predict")
    st.success("The Predicted Percentage is 100.00")
else:
    st.button("Predict")
    st.success("The Predicted Percentage is {}".format(prediction))
 