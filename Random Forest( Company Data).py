#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd


# In[21]:


company=pd.read_csv('C:\\Users\\hp\\Downloads\\Company_Data.csv')


# In[22]:


company.columns


# In[23]:


company.isna().sum()


# In[24]:


company.Sales.median()


# In[25]:


#create bins for sales
cut_labels=['low','medium','high']
cut_bins=[-1,5.66,12,17]
company['sales']=pd.cut(company['Sales'],bins=cut_bins,labels=cut_labels)
company.pop('Sales')


# In[26]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
company['ShelveLoc']=label_encoder.fit_transform(company['ShelveLoc'])
company['Urban']=label_encoder.fit_transform(company['Urban'])
company['US']=label_encoder.fit_transform(company['US'])


# In[27]:


array=company.values
X=array[:,0:10]
Y=array[:,10]
#splitting data using K-Fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold=KFold(n_splits=10,random_state=7)


# In[28]:


#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,max_features=3)
results=cross_val_score(model,X,Y,cv=kfold)
print(results.mean())#73.75


# In[ ]:




