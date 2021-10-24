#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[15]:


fraud=pd.read_csv('C:\\Users\\hp\\Downloads\\Fraud_check.csv')


# In[16]:


fraud.columns


# In[17]:


fraud.columns=['under_grad','marital_status','taxable_income','city_pop','work_exp','urban']


# In[18]:


#creating bins for taxable_income=>to categorical
cut_labels=['Risky','Good']
cut_bins=[0,30000,99620]


# In[19]:


fraud['tax_inc']=pd.cut(fraud['taxable_income'],bins=cut_bins,labels=cut_labels)
fraud.pop('taxable_income')


# In[20]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
fraud['under_grad']=label_encoder.fit_transform(fraud['under_grad'])
fraud['marital_status']=label_encoder.fit_transform(fraud['marital_status'])
fraud['urban']=label_encoder.fit_transform(fraud['urban'])


# In[21]:


col_names=list(fraud.columns)
predictors=col_names[0:5]
target=col_names[5]


# In[22]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(fraud,test_size=0.3,random_state=0)


# In[23]:


from sklearn.tree import DecisionTreeClassifier as DS
model=DS(criterion='entropy')
model.fit(train[predictors],train[target])
train_pred=model.predict(train[predictors])
test_pred=model.predict(test[predictors])


# In[24]:


train_acc=np.mean(train_pred==train[target])
test_acc=np.mean(test_pred==test[target])


# In[25]:


train_acc#1.0


# In[26]:


test_acc#0.53


# In[ ]:




