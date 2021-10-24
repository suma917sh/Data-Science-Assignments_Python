#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#Plot Tools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#Model Building
from sklearn.preprocessing import StandardScaler
import sklearn
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import InputLayer,Dense
import tensorflow as tf
#Model Validation
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error


# In[3]:


data=pd.read_csv('C:\\Users\\hp\\Downloads\\gas_turbines.csv')
data


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


X = data.loc[:,['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP', 'CO','NOX']]
y= data.loc[:,['TEY']]


# In[7]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)


# In[8]:


def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[9]:


estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=100, verbose=False)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# In[10]:


estimator.fit(X, y)
prediction = estimator.predict(X)


# In[11]:


prediction


# In[12]:


a=scaler.inverse_transform(prediction)
a


# In[13]:


b=scaler.inverse_transform(y)
b


# In[14]:


mean_squared_error(b,a)


# In[15]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[16]:


estimator.fit(X_train, y_train)
prediction = estimator.predict(X_test)


# In[17]:


prediction


# In[18]:


c=scaler.inverse_transform(prediction)


# In[19]:


d=scaler.inverse_transform(y_test)


# In[20]:


mean_squared_error(d,c)


# In[ ]:




