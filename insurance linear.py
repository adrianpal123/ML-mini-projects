import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot


data=pd.read_csv("insurance.csv",sep=",")

data=data[["bmi","age","children","charges"]]
print(data.head())

predict='charges'

X=np.array(data.drop([predict],1))
Y=np.array(data[predict])

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

linear=linear_model.LinearRegression()
linear.fit(x_train,y_train)
acc=linear.score(x_test,y_test)
print(acc)