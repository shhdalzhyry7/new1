import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('economic_data.csv')
print(data)

print(data.describe())


print(data.head(4))

plt.scatter(data['Year'], data['GDP'])
plt.show()

#y=mx+b

print(data.head())

x=data.iloc[ : ,  :1]
y=data.iloc[ : ,1]

print(x)
print(y)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

#m
print(model.coef_)
#b
print(model.intercept_)


#GDP(y)=model.coef_ * Year(x) + model.intercept_
plt.plot(x,model.predict(x))
plt.scatter(x, y,color='red')




model.predict([[2035]])
model.predict([[2040]])
model.predict([[2050]])

#accuracies   85% الدقة
model.score(x, y)
