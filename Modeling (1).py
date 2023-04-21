#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as nump
import pandas as pan
import matplotlib.pyplot as plt
import seaborn as seans
import mpl_toolkits
get_ipython().run_line_magic('matplotlib', 'inline')

df = pan.read_excel(r"C:\Users\kalaz\lastdf.xlsx")
df


# In[2]:


df.describe()


# In[3]:


seans.displot(df['price']);


# In[4]:


df.hist(bins=70, figsize=(20,20))
plt.show()


# In[5]:


plt.figure(figsize=(20,20))
seans.jointplot(x=df.lat.values, y=df.long.values, height=15)
plt.ylabel('Longitude')
plt.xlabel('Latitude')
plt.show()


# In[6]:


df['bedrooms'].value_counts().plot(kind = 'bar')
plt.title('Number of bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Properties')


# In[7]:


plt.scatter(df['price'],(df['sqft_living']+df['sqft_lot']))
plt.xlabel('Price')
plt.ylabel('Total square footage')


# In[8]:


plt.scatter(df.price,df.floors)
plt.xlabel('Price')
plt.ylabel('Number of floors')


# In[9]:


plt.scatter(df.price,df.grade)
plt.xlabel('Price')
plt.ylabel('Grade')


# In[10]:


plt.scatter(df.price,df.zipcode)
plt.xlabel('Price')
plt.ylabel('Zipcode')


# In[11]:


train = df.drop(['id','price'],axis=1)
train


# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


# In[21]:


x = df[['zipcode','bedrooms','grade','sqft_living']]
y=df['price']


# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)


# In[25]:


reg = LinearRegression()
reg.fit(x_train, y_train)


# In[26]:


print(reg.coef_)


# In[28]:


y_pred = reg.predict(x_test)


# In[29]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression")
plt.show()


# In[30]:


plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', lw=2)
plt.plot(y_test, y_pred, '.', alpha=0.5)


# In[32]:


corr_coef = nump.corrcoef(y_test, y_pred)[0, 1]
print("Correlation coefficient:", corr_coef)


# In[35]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title("Scatter plot of predicted vs actual prices")
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")


# In[39]:


plt.plot(y_test, y_test, color='red')
plt.plot(nump.unique(y_test), nump.poly1d(nump.polyfit(y_test, y_pred, 1))(nump.unique(y_test)), color='blue')
plt.show()


# In[40]:


import joblib


# In[42]:


x = df[['bedrooms','zipcode','sqft_living']]
y = df ['price']
reg = LinearRegression()
reg.fit(x,y)


# In[43]:


joblib.dump(reg, 'model.joblib')


# In[ ]:




