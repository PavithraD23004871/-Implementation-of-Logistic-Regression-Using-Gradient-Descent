# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn
4.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
 

## Program:
/*
Program to implement the the Logistic Regression Using Gradient Descent.

Developed by: Pavithra D
RegisterNumber:  212223230146

*/
```

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df=pd.read_csv('/content/Mammal_Cart.csv')
data=df.copy()
data.describe()

label_encoder=LabelEncoder()
data['Toothed']=label_encoder.fit_transform(data['Toothed'])
data['Hair']=label_encoder.fit_transform(data['Hair'])
data['Breathes']=label_encoder.fit_transform(data['Breathes'])
data['Legs']=label_encoder.fit_transform(data['Legs'])
data['Species']=label_encoder.fit_transform(data['Species'])

x=data.drop('Species',axis=1)
y=data['Species']

clf=DecisionTreeClassifier(criterion='gini')

clf.fit(x,y)
plt.figure(figsize=(18,6))
plot_tree(clf,feature_names=x.columns,class_names=['Reptile','Mammal'],filled=True)
plt.show()
```


## Output:
1.HEAD

![image](https://github.com/PavithraD23004871/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138955967/ec735e8d-fcf7-4c49-adf7-511400003415)

![image](https://github.com/PavithraD23004871/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138955967/125cbd8e-498d-4ccd-8476-a4ecaad83996)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

