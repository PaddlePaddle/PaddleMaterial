#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np

data = pd.read_csv('AccuracyTest.csv')
data


# In[37]:


y_true = data.iloc[:, 6]
y_true


# In[38]:


y_pred = data.iloc[:, 7]
y_pred


# In[39]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)


# In[40]:


print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


# In[ ]:




