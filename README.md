# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM :
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required :
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program :
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Dharini PV
RegisterNumber: 212222240024
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output :
## df.head()
![image](https://github.com/DHARINIPV/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119400845/9dc0a94e-56c5-4365-880e-8bf693eccc0c)


## df.tail()
![image](https://github.com/DHARINIPV/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119400845/448886dc-9637-4b5b-b02c-c9c0b6c52dc1)


## Array value of X
![image](https://github.com/DHARINIPV/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119400845/02ee8275-24e1-43a7-b890-dca33c79417a)


## Array value of Y
![image](https://github.com/DHARINIPV/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119400845/a8ef54f9-4122-4eda-99fc-f7acd429fcff)


## Values of Y prediction
![image](https://github.com/DHARINIPV/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119400845/d6e60c59-1a76-4b31-b655-6783bfab4ea4)


## Array values of Y test
![image](https://github.com/DHARINIPV/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119400845/b0415a2e-e44c-4892-8aec-4415485a54c1)


## Training Set Graph
![image](https://github.com/DHARINIPV/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119400845/2759bf17-0511-489c-abec-266fc884934c)


## Test Set Graph
![image](https://github.com/DHARINIPV/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119400845/60cf18dc-4a7d-43ff-837a-8b29d5ef6d01)


## Values of MSE, MAE and RMSE
![image](https://github.com/DHARINIPV/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119400845/a94c7007-1472-4f74-a0a1-b250bd0c204f)


## Result :
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
