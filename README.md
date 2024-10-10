# Home-Price-Prediction

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt



df=pd.read_excel("/content/home price.xlsx")
df


%matplotlib inline
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')

new_df=df.drop('price',axis='columns')
new_df

model=linear_model.LinearRegression()
model.fit(new_df,df.price)

 model.predict([[5000]])

model.coef_

model.intercept_

135.71917808*5000+181041.09589041077

df=pd.read_excel("/content/area.xlsx")
df

p=model.predict(df)
df['area']

p

df=pd.read_excel("/content/area.xlsx")
df

df=pd.read_excel("/content/area.xlsx")
df

df=pd.read_excel("/content/linear regression.xlsx")
df

import math
df.bedrooms.median()

df.bedrooms=df.bedrooms.fillna(df.bedrooms.median())
df

model=linear_model.LinearRegression()
model.fit(df.drop('price',axis='columns'),df.price)


model.predict([[3000,3,40]])

model.coef_

model.intercept_

df=pd.read_excel("/content/linear regression.xlsx")
df

df=pd.read_excel("/content/LR.xlsx")
df

import math
df.bedrooms.median()

df.bedrooms=df.bedrooms.fillna(df.bedrooms.median())
df.bedrooms

df.bedrooms.mean
df.bedrooms=df.bedrooms.fillna(df.bedrooms.mean())
df.bedrooms

model=linear_model.LinearRegression()
model.fit(df.drop('price',axis=1),df.price)

model.predict([[6000,7,35]])

model.coef_

model.intercept_

 93.37705325*6000+ 31305.99561341*7+ -2068.79500177*35+226819.46740530716

df=pd.read_excel("/content/logistic r.xlsx")
df

%matplotlib inline
plt.xlabel('age')
plt.ylabel('bought_insurance')
plt.scatter(df.age,df.bought_insurance, marker='*',color='blue')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(df[['age']],df.bought_insurance,test_size=0.3)

len(x_train)

len(y_train)

len(x_test)

x_train

x_test

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

model.predict(x_test)

model.predict_proba(x_test)

model.score(x_test,y_test)
