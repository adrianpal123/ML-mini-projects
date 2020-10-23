import  numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model




DATA = pd.read_csv("Fish.csv",sep=",")
DATA=DATA[["Length1","Length2","Length3","Height","Width","Weight"]]

print(DATA.head())


PREDICT="Weight"

X= np.array(DATA.drop([PREDICT],1))
Y=np.array(DATA[PREDICT])

print()
print(X)
print()
print(Y)

X_train, X_test, Y_train, Y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.1)
''''
best=0
i=0
while i<30:
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(X_train, Y_train)
    acc = linear.score(X_test, Y_test)

    print(acc)
    with open("Fish.pickle", "wb") as f:
        pickle.dump(linear, f)
    if best<acc:
        best=acc
    i+=1

print()
print(best)
'''

pickle_in=open("Fish.pickle","rb")
linear=pickle.load(pickle_in)

print("Co: \n", linear.coef_)
print("Intercept: \n",linear.intercept_)

predictions=linear.predict(X_test)

for i in range(len(predictions)):
    print(predictions[i],X_test[i],X_train[i])
    print(Y_test[i])

p="Length3"
style.use("ggplot")
pyplot.scatter(DATA[p],DATA["Weight"])
pyplot.xlabel(p)
pyplot.ylabel("Weight")
pyplot.show()
