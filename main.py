# Importing required libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Reading Dataset

df = pd.read_csv("https://raw.githubusercontent.com/SoleCodr/Prediction-of-Marks-scored-by-student/master/Data.txt?token=AGEY7MBIGAHSIIG6U7BN7ES7FWEOO")

# Preparing the Data

attr = df.iloc[:,:-1].values
labels = df.iloc[:,1].values

# Splitting data for training and testing 

X_train, X_test, y_train, y_test = train_test_split(attr, labels, test_size=0.2, random_state=0) 

# Training the Algorithm

LR = LinearRegression()
LR.fit(X_train, y_train)

# pickling file

pickle.dump(LR,open("Linear_Reg.pkl","wb"))
