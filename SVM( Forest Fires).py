#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[4]:


forest_fire=pd.read_csv('C:\\Users\\hp\\Downloads\\forestfires.csv')


# In[6]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
forest_fire['month']=label_encoder.fit_transform(forest_fire['month'])
forest_fire['day']=label_encoder.fit_transform(forest_fire['day'])


# In[8]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(forest_fire,test_size=0.3,random_state=0)
columns=list(forest_fire.columns)
predictors=columns[0:30]
target=columns[30]


# In[9]:


#SVM Classification using Kernels: linear,poly,rbf


# In[10]:


from sklearn.svm import SVC

#kernel=linear
model_linear=SVC(kernel='linear')
model_linear.fit(train[predictors],train[target])
train_pred_linear=model_linear.predict(train[predictors])
test_pred_linear=model_linear.predict(test[predictors])
train_lin_acc=np.mean(train_pred_linear==train[target])
test_lin_acc=np.mean(test_pred_linear==test[target])
train_lin_acc#1.0
test_lin_acc#0.96


# In[11]:


#kernel=poly
model_poly=SVC(kernel='poly')
model_poly.fit(train[predictors],train[target])
train_pred_poly=model_poly.predict(train[predictors])
test_pred_poly=model_poly.predict(test[predictors])
train_poly_acc=np.mean(train_pred_poly==train[target])
test_poly_acc=np.mean(test_pred_poly==test[target])
train_poly_acc#0.76
test_poly_acc#0.75


# In[12]:


#kernel=rbf
model_rbf=SVC(kernel='rbf')
model_rbf.fit(train[predictors],train[target])
train_pred_rbf=model_rbf.predict(train[predictors])
test_pred_rbf=model_rbf.predict(test[predictors])
train_rbf_acc=np.mean(train_pred_rbf==train[target])
test_rbf_acc=np.mean(test_pred_rbf==test[target])
train_rbf_acc#0.76
test_rbf_acc#0.72


# In[ ]:




