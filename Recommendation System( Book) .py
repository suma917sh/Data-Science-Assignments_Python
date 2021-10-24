#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd


# In[11]:


book_rating_data=pd.read_csv('C:\\Users\\hp\\Downloads\\book.csv',encoding='ISO-8859-1')


# In[12]:


book_rating_data.columns


# In[13]:


book_rating_data.columns=['sr_no','user_id','book_title','book_rating']


# In[14]:


book_rating_data.head()


# In[19]:


len(book_rating_data.user_id.unique())


# In[18]:


len(book_rating_data.book_title.unique())


# In[20]:


book_rating_data_matrix=book_rating_data.pivot_table(index='user_id',columns='book_title',values='book_rating').reset_index(drop=True)
book_rating_data_matrix


# In[21]:


book_rating_data_matrix.index=book_rating_data.user_id.unique()
book_rating_data_matrix.fillna(0,inplace=True)


# In[22]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
user_similarity=1-pairwise_distances(book_rating_data_matrix.values,metric='cosine')
user_similarity


# In[23]:


#store results in dataframe
user_similarity_df=pd.DataFrame(user_similarity)


# In[24]:


#set index and column names to userid's
user_similarity_df.index=book_rating_data.user_id.unique()
user_similarity_df.columns=book_rating_data.user_id.unique()
user_similarity_df.iloc[0:5,0:5]
np.fill_diagonal(user_similarity,0)
user_similarity_df.iloc[0:5,0:5]


# In[25]:


#most similar user
user_similarity_df.idxmax(axis=1)[0:5]


# In[26]:


#recommended books for userid==276726
user1=book_rating_data[book_rating_data['user_id']==276726]
user1.book_title


# In[ ]:




