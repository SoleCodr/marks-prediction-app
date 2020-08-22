import streamlit as st
import numpy as np
import pickle

Linear_Reg = pickle.load(open("Linear_Reg.pkl","rb"))

def pred(hours):
    val=np.array([[hours]]).astype(np.float64)
    predection=Linear_Reg.predict(val)
    return float(predection)

def main():
    st.title(" Score Prediction ")
    hours=st.text_input("Hours Of Study")
    if float(hours) <= 0 or float(hours) >24:
        st.error("Hours should be in between 0-24hr")
    else:
        if st.button("Predict"):
            output=pred(hours)
            st.success("The Predicted Percentage is {}".format(output))


if  __name__=="__main__":
    main()
