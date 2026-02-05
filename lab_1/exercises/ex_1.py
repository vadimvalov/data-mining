# Exercise 1.3 (1) Show the confusion matrix

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# be aware, install certificates, use command:
# open "/Applications/Python 3.13/Install Certificates.command"

# csv of the wine quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(url, sep=';')
# print(df.head())  ### uncomment and check the data is loaded properly


X = df.drop(columns=['quality'])
y = df['quality']

# where X is the features and y is the target variable
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model

# increase the number of iterations to avoid 
# "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT" error

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# predict the test set
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# OUTPUT:

# Confusion matrix:
# [[ 0  0  1  0  0  0]
#  [ 0  0  8  2  0  0]
#  [ 0  0 98 31  1  0]
#  [ 0  0 47 79  6  0]
#  [ 0  0  3 33  6  0]
#  [ 0  0  0  1  4  0]]