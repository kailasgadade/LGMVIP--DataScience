#!/usr/bin/env python
# coding: utf-8

# # LetsGrow More: Data Science Internship
# ## Task 1 : Iris flowers classification ML project
# ### Intern name : Gadade Kailas Rayappa
# ### Step 1 : Importing Libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# ### Step 2 : Importing Dataset

# In[2]:


data=pd.read_csv("C:\\Users\\Kailas\\OneDrive\\Desktop\\Iris.csv")
data


# In[3]:


data.head()# for first 5 rows


# In[4]:


data.tail()# for last 5 rows


# In[5]:


data.shape# Dimesnions of dataset


# In[6]:


data.describe()# Discriptive Statistics


# In[7]:


data.info()# information of datasets


# ### In this iris dataset have one target column which is "Species" it is categorical. It has 4 variables of iris and none of the cells have null values.

# ### Step 2 : Exploratory Data Analysis

# In[8]:


data['Species'].unique()


# In[9]:


plt.close()
sns.set_style('whitegrid')
sns.pairplot(data,hue='Species',height=3);
plt.show()


# In[10]:


data.corr()


# #### Using heatmap to know the correlation among the variables

# In[11]:


plt.figure(figsize=(12,5))
sns.heatmap(data.corr(),annot=True,cmap='gnuplot2')
plt.show()


# #### From above heatmap it is seen that petal length and petal width are highly correlated.

# In[12]:


plt.figure(figsize=(8,6))
sns.boxplot(x="Species",y="SepalLengthCm",data=data)
plt.show()


# In[13]:


plt.subplot(2,2,1)
sns.boxplot(x= data.Species,y=data.SepalLengthCm)
plt.subplot(2,2,2)
sns.boxplot(x= data.Species,y=data.SepalWidthCm)
plt.subplot(2,2,3)
sns.boxplot(x= data.Species,y=data.PetalLengthCm)
plt.subplot(2,2,4)
sns.boxplot(x= data.Species,y=data.PetalWidthCm)
plt.show()


# In[14]:


# Removing the 'Id' column
data.drop('Id' , axis=1, inplace=True) 
data.head()
data.groupby('Species').mean().plot.bar()
plt.show()


# In[15]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.distplot(data['PetalLengthCm'])
plt.subplot(2,2,2)
sns.distplot(data['PetalWidthCm'])
plt.subplot(2,2,3)
sns.distplot(data['SepalLengthCm'])
plt.subplot(2,2,4)
sns.distplot(data['SepalWidthCm'])
plt.show()


# ##### From above boxplot in "Iris-virginica" one outlier is detected.

# In[17]:


sns.pairplot(data.corr())
plt.show()


# ### Step 3 : Building ML model for classification

# In[18]:


X = data[['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['SepalLengthCm']


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.65, random_state=101)
from sklearn.linear_model import LinearRegression
df = LinearRegression()
df.fit(X_train, y_train)
predictions = df.predict(X_test)


# In[20]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))


# In[21]:


print('coefficient of determination:', df.score(X_train,y_train))
print('intercept:', df.intercept_)
print('slope:', df.coef_) 


# In[22]:


df.predict([[5.1, 2.5, 1.1]])


# In[23]:


df.predict([[7.5, 3.0, 1.8]])


# In[24]:


df.predict([[4.6, 3.5, 0.2]])


# ### Thank you ...
