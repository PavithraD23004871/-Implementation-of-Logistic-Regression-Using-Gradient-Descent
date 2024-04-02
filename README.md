# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas. 2.Calculate the null values from dataframe and apply label encoder. 3.Apply decision tree classifier on the dataframe. 4.obtain the value of accuracy and data prediction.
 

## Program:
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Pavithra D
RegisterNumber:  212223230146
*/
```
import pandas as pd
data=pd.read_csv('/content/Employee_EX6.csv')
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

## Output:
1.HEAD
![image](https://github.com/PavithraD23004871/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138955967/b1461145-e128-454e-9315-a22ca425e739)

ACCURACY

![image](https://github.com/PavithraD23004871/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138955967/301d9672-55a5-4f65-bd05-2932111805b2)

![image](https://github.com/PavithraD23004871/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138955967/0133f8b1-0bd1-49cb-bd41-82d97c1b05bf)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

