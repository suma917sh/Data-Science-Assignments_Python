#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


company=pd.read_csv('C:\\Users\\hp\\Downloads\\Company_Data.csv')


# In[3]:


company.columns


# In[4]:


company.Sales.median()


# In[5]:


company.isna().sum()


# In[6]:


#create bins for sales
cut_labels=['Low','Medium','High']
cut_bins=[-1,5.66,12,17]


# In[7]:


company['sales']=pd.cut(company['Sales'],labels=cut_labels,bins=cut_bins)


# In[8]:


company.pop('Sales')


# In[11]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
company['ShelveLoc']=label_encoder.fit_transform(company['ShelveLoc'])
company['Urban']=label_encoder.fit_transform(company['Urban'])
company['US']=label_encoder.fit_transform(company['US'])


# In[13]:


col_names=list(company.columns)
predictors=col_names[0:10]
target=col_names[10]


# In[15]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(company,test_size=0.3,random_state=0)


# In[16]:


from sklearn.tree import DecisionTreeClassifier as DS
model=DS(criterion='entropy')
model.fit(train[predictors],train[target])


# In[17]:


train_pred=model.predict(train[predictors])
test_pred=model.predict(test[predictors])


# In[18]:


train_acc=np.mean(train_pred==train[target])
test_acc=np.mean(test_pred==test[target])


# In[19]:


train_acc#1.0


# In[20]:


test_acc#0.66


# In[ ]:




