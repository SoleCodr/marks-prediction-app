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
    st.write('''
        **Assuming the study hours to be maximum 10 hours.**
        ''')
    
    hours=st.number_input("Enter Hours Of Study")
    if float(hours) <= 0.00 or float(hours) >10.00:
        st.error("Hours should be in between 0-10hr")
    elif float(hours)>9.88:
        if st.button("Predict"):
            st.success("The Predicted Percentage is 100.00 ")
    else:
        if st.button("Predict"):
            output=pred(hours)
            st.success("The Predicted Percentage is {}".format(output))


if  __name__=="__main__":
    main()
