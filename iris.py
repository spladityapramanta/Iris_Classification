# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:59:09 2017

@author: SPL Aditya Pramanta
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

#Importing data sets
dataset = pd.read_csv("bezdekIris.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

# Taking care of the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
onehotencoder = OneHotEncoder(categorical_features=[0])
# Y = onehotencoder.fit_transform(Y).toarray()

# Spliting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, test_size = 0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_Train = Sc_X.fit_transform(X_Train)
X_Test = Sc_X.transform(X_Test)
