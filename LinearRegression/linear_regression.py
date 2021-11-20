# import libraries
import pandas as pd # data manipulation
import numpy as np
import matplotlib.pyplot as plt # to view data
from sklearn.linear_model import LinearRegression # basic machine learning algorithm

# import data and view it
df = pd.read_csv('homeprices.csv')
print(df)

# fill in the missing values with the median
import math
# we use floor because there has to be an integer number of bedrooms
median_bedrooms = math.floor(df.bedrooms.median()) 
print("Median bedrooms: {}".format(median_bedrooms))

# create the model and fit it
reg = LinearRegression()
# the first parameter is the x variable
# the second parameter is the y variable (the one we are trying to predict)
reg.fit(df[['area']], df.price)

# print the coefficient and intercept
# the equation is linear, so it is in the form y=ax+b
print(reg.coef_) # this is a in the equation
print(reg.intercept_) # this is b in the equation

# visualize the data and linear regression
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df['area'], df['price']) # trying to plot the data
x = np.linspace(df.area.min()-100, df.area.max()+100, endpoint=True)
y = reg.coef_ * x + reg.intercept_
plt.plot(x, y, color='r')
plt.show()

# import data to test the model on
df_areas = pd.read_csv('areas.csv')
print(df_areas)

# make predictions and view them
y_hat = reg.predict(df_areas)
print(y_hat)

# add predictions to the DataFrame
df_areas['price'] = y_hat
print(df_areas)
