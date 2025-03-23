#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


# ## EDA (Exploratory Data Analysis)

# In[2]:


df=pd.read_csv('train.csv')


# In[3]:


df.head()


# In[4]:

# FEATURE EXTRACTION

# In[71]:


shrink = df.copy()


# In[72]:


cols = ["LotArea", "BedroomAbvGr", "FullBath", "MSSubClass", "PavedDrive", "PoolArea", "PoolQC", "BsmtQual", "Heating", "CentralAir", "MSZoning", "GarageCars"]

for col in list(shrink):
    if col not in cols:
        shrink.drop([col], axis=1, inplace=True)


# In[73]:


one = [20,30,40,45,50,120,150]
two = [60,70,75,160]
multi = [80,85,90,180,190]

def stories_num(num):
    if num in one:
        return 'one'
    elif num in two:
        return 'two'
    else:
        return 'multi'


# In[74]:


buff = shrink['MSSubClass']
stories = buff.map(stories_num)
shrink['stories'] = stories
del buff
del stories
shrink.drop(['MSSubClass'], axis=1, inplace=True)
shrink['SalePrice'] = df['SalePrice']


# In[75]:


shrink[shrink['MSZoning'] == 'C (all)']


# In[76]:


shrink['MSZoning'].value_counts()


# In[77]:


rl = list(shrink[shrink['MSZoning'] == 'RL'].SalePrice.values)
rm = list(shrink[shrink['MSZoning'] == 'RM'].SalePrice.values)
fv = list(shrink[shrink['MSZoning'] == 'FV'].SalePrice.values)
rh = list(shrink[shrink['MSZoning'] == 'RH'].SalePrice.values)
c = list(shrink[shrink['MSZoning'] == 'C (all)'].SalePrice.values)
rl = np.array(rl)
rm = np.array(rm)
fv = np.array(fv)
rh = np.array(rh)
c = np.array(c)


# In[78]:


print(rl.mean())
print(rm.mean())
print(fv.mean())
print(rh.mean())
print(c.mean())


# In[79]:


def area(x):
    if x == 'RH':
        return 'RM'
    elif x == 'RL':
        return 'RL'
    elif x == 'FV':
        return 'FV'
    elif x == 'C (all)':
        return 'C'
    
area_col = shrink['MSZoning']
temp = area_col.map(area)
shrink['Area'] = temp
del temp
del area_col
shrink.drop(['MSZoning'],axis=1, inplace=True)


# In[80]:



shrink["Area"].value_counts()


# In[83]:


def is_pool(x):
    if x == 0:
        return 'no'
    else:
        return 'yes'
    
pool_col = shrink['PoolArea']
temp = pool_col.map(is_pool)
shrink['Pool'] = temp
del pool_col
del temp
shrink.drop(['PoolArea'], axis =1, inplace = True)


# In[84]:


def pavement(x):
    if x == 'Y':
        return 'Y'
    elif x == 'N':
        return 'N'
    elif x == 'P':
        return 'Y'
    
pave_col = shrink['PavedDrive']
temp = pave_col.map(pavement)
shrink['Pavement'] = temp
del pave_col
del temp
shrink.drop(['PavedDrive'], axis =1, inplace = True)


# In[85]:


shrink.BsmtQual.value_counts()


# In[86]:


ta = list(shrink[shrink['BsmtQual'] == 'TA'].SalePrice.values)
gd = list(shrink[shrink['BsmtQual'] == 'Gd'].SalePrice.values)
ex = list(shrink[shrink['BsmtQual'] == 'Ex'].SalePrice.values)
fa = list(shrink[shrink['BsmtQual'] == 'Fa'].SalePrice.values)

ta = np.array(ta)
gd = np.array(gd)
ex = np.array(ex)
fa = np.array(fa)

print(fa.mean())
print(ta.mean())
print(gd.mean())
print(ex.mean())


# In[87]:


def base_qual(x):
    if x == 'Fa':
        return '1'
    elif x == 'TA':
        return '2'
    elif x == 'Gd':
        return '3'
    else:
        return '4'
    
base_col = shrink['BsmtQual']
temp = base_col.map(base_qual)
shrink['Basement'] = temp
del base_col
del temp
shrink.drop(['BsmtQual'], axis =1, inplace = True)


# In[88]:


shrink['Basement'] = shrink['Basement'].astype(int)


# In[89]:


def story(x):
    if x == 'one':
        return '1'
    elif x == 'two':
        return '2'
    elif x == 'multi':
        return '3'
    
story_col = shrink['stories']
temp = story_col.map(story)
shrink['Stories'] = temp
del story_col
del temp
shrink.drop(['stories'], axis =1, inplace = True)


# In[90]:


shrink.CentralAir.value_counts()


# In[91]:


shrink.head()


# In[92]:


cleansed = shrink.copy()


# In[93]:


cleansed


# In[94]:


encoded = cleansed.copy()


# In[95]:


encoded['Pavement'] = np.where(encoded['Pavement'].str.contains("Y"),1,0)
encoded['CentralAir'] = np.where(encoded['CentralAir'].str.contains("Y"),1,0)
encoded['Pool'] = np.where(encoded['Pool'].str.contains("yes"),1,0)


# In[96]:


encoded.head()


# In[97]:


dum_df = pd.get_dummies(encoded, columns=["Area", "Heating"], prefix=["Area", "Heating"])
# merge with main df bridge_df on key values
# encoded = encoded.join(dum_df)
dum_df


# In[98]:


print(cleansed.shape)
print(dum_df.shape)


# In[99]:


x = dum_df.drop(['SalePrice'], axis = 1)
y = dum_df['SalePrice']


# In[100]:


for col in list(x):
    x[col] = pd.to_numeric(x[col])


# In[102]:


x.drop(['Area_C'], inplace = True, axis = 1)


# In[104]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[105]:


encoded["Area"] = encoded["Area"].astype(str)


# In[108]:


le = LabelEncoder()
le.fit(encoded["Area"])
le_enc = le.transform(encoded["Area"])


# In[109]:


encoded["Area"].value_counts()


# In[110]:


le_enc = le_enc.reshape(len(le_enc), 1)


# In[111]:


ohe = OneHotEncoder(drop='first')
ohe.fit(le_enc)
encoded["Area"] = ohe.transform(le_enc)


# In[112]:


x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[113]:


regressor_rf = RandomForestRegressor(random_state = 23)
regressor_rf.fit(x_train, y_train)


# In[114]:


y_pred_rf = regressor_rf.predict(x_test)


# In[116]:
pickle.dump(regressor_rf,open('Random_model.pkl','wb'))


# In[ ]:
#load model from the disk
Random_model=pickle.load(open('Random_model.pkl','rb'))
