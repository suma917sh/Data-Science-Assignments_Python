#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd


# In[13]:


fraud=pd.read_csv('C:\\Users\\hp\\Downloads\\Fraud_check.csv')


# In[14]:


fraud.columns


# In[15]:


fraud.columns=['under_grad','marital_status','taxable_income','city_pop','work_exp','urban']


# In[16]:


#creating bins for taxable_income=>to categorical
cut_labels=['Risky','Good']
cut_bins=[0,30000,99620]
fraud['tax_inc']=pd.cut(fraud['taxable_income'],bins=cut_bins,labels=cut_labels)
fraud.pop('taxable_income')


# In[17]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
fraud['under_grad']=label_encoder.fit_transform(fraud['under_grad'])
fraud['marital_status']=label_encoder.fit_transform(fraud['marital_status'])
fraud['urban']=label_encoder.fit_transform(fraud['urban'])


# In[18]:


array=fraud.values
X=array[:,0:5]
Y=array[:,5]
#splitting data using K-Fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold=KFold(n_splits=10,random_state=7)


# In[19]:


#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,max_features=3)
results=cross_val_score(model,X,Y,cv=kfold)
print(results.mean())#74.66


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




