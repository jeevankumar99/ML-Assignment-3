from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n----------- HOUSING PRICE PREDICTOR ------------\n")
data = load_boston()

array = data.feature_names
print(array)
array = np.append(array,['medv'])

data, target = data.data, data.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,target,test_size=0.3)

print(Xtrain.shape,Ytrain.shape)
print(Xtest.shape,Ytest.shape)

lin_model = LinearRegression()
lin_model.fit(Xtrain,Ytrain)
Ytrain_predict = lin_model.predict(Xtrain)

rmse = (np.sqrt(mean_squared_error(Ytrain,Ytrain_predict)))
r2 = r2_score(Ytrain,Ytrain_predict)

print("Model performance for training set is :\n ")
print("Root Mean Square Error: ",rmse,"\n")
print("R2 sore is: ",r2,"\n")

Ytest_predict = lin_model.predict(Xtest)

rmse = (np.sqrt(mean_squared_error(Ytest,Ytest_predict)))
r2 = r2_score(Ytest,Ytest_predict)

print("Model performance for testing set is :\n ")
print("Root Mean Square Error: ",rmse,"\n")
print("R2 sore is: ",r2,"\n")


plt.scatter(Ytest,Ytest_predict,c = 'green')
plt.xlabel("Price in $1000's")
plt.ylabel("Predicted value")
plt.title("True value vs predicted value: Linear Regression")
plt.show()