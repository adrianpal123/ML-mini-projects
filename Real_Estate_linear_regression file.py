import  numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model
import csv

data=pd.read_csv("Real estate.csv", sep=",")

data=data[["X2 house age","X3 distance to the nearest MRT station","X4 number of convenience stores","Y house price of unit area"]]

print(data.head())






predict="Y house price of unit area"

X=np.array(data.drop([predict],1))
Y=np.array(data[predict])

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

'''best = 0
i = 0
while i<10000:

    linear=linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc=linear.score(x_test,y_test)


    if best < acc:
        best = acc
        with open("Real estate.pickle", "wb") as f:
            pickle.dump(linear, f)

    i+=1


'''

pickle_in=open("Real estate.pickle","rb")
linear=pickle.load(pickle_in)



p="X4 number of convenience stores"
style.use("ggplot")
pyplot.scatter(data[p],data["Y house price of unit area"])
pyplot.xlabel(p)
pyplot.ylabel("Y house price of unit area")
pyplot.show()


