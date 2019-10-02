import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("Air Quality123.csv")
print(df)
x = df[['PM2.5','PM10','NO2','NH3','SO2','CO','OZONE']]
y = df['AirQualityIndex']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
#random_state=10
reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)
print(reg.predict(x_test))
print(y_test)
print(reg.score(x_test,y_test)*100,"%")
print(x_test)
print(reg.predict([[0,0,0,0,0,0,0]]))
'''
plt.scatter(df['PM2.5'],df['AirQualityIndex'])
plt.show()
plt.scatter(df['PM10'],df['AirQualityIndex'])
plt.show()
plt.scatter(df['NO2'],df['AirQualityIndex'])
plt.show()
plt.scatter(df['NH3'],df['AirQualityIndex'])
plt.show()
plt.scatter(df['SO2'],df['AirQualityIndex'])
plt.show()
plt.scatter(df['CO'],df['AirQualityIndex'])
plt.show()
plt.scatter(df['OZONE'],df['AirQualityIndex'])
plt.show()
'''
