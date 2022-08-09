#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd 
from IPython.display import HTML


# In[2]:


# importing dataset

dataset = pd.read_csv('weatherAUS.csv')

x = dataset.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values #independent variable

y = dataset.iloc[:,-1].values # dependent variable


# In[3]:


y = y.reshape(-1,1)  # 1D list to 2D list
y


# In[4]:


# dealing with invalid data
# Null values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
x = imputer.fit_transform(x)
y = imputer.fit_transform(y)


# In[5]:


# ENCODING DATASET
# encoding charaters into numbers

from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
x[:,0] = le1.fit_transform(x[:,0])

le2 = LabelEncoder()
x[:,4] = le2.fit_transform(x[:,4])

le3 = LabelEncoder()
x[:,6] = le3.fit_transform(x[:,6])

le4 = LabelEncoder()
x[:,7] = le4.fit_transform(x[:,7])

le5 = LabelEncoder()
x[:,-1] = le5.fit_transform(x[:,-1])

le6 = LabelEncoder()
y[:,-1] = le6.fit_transform(y[:,-1])


# In[6]:


x


# In[7]:


y = np.array(y,dtype=float)
y


# In[9]:


# Feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(x)


# In[10]:


# Splitting data into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)


# In[11]:


y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


# In[12]:


x_train


# In[13]:


y_train


# In[14]:


# Training model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 100,random_state=0)
classifier.fit(x_train,y_train) 


# In[15]:


classifier.score(x_train,y_train)   # accuracy of model on training data


# In[16]:


y_pred = le6.inverse_transform(np.array(classifier.predict(x_test),dtype=int))
y_test = le6.inverse_transform(np.array(y_test,dtype=int))


# In[17]:


y_pred = y_pred.reshape(-1,1)
y_test = y_test.reshape(-1,1)


# In[18]:


df = np.concatenate((y_test,y_pred),axis=1)
dataframe = pd.DataFrame(df,columns=['Will it Rain tomorrow?','Preditiction'])


# In[19]:


dataframe


# In[20]:


from sklearn.metrics import accuracy_score  # tesing accuracy of model on test data
accuracy_score(y_test,y_pred)


# In[21]:


# this step is only important for opening our result dataframe in new window 

def View(df):        
    css = """<style>
    table { border-collapse: collapse; border: 3px solid #eee; }
    table tr th:first-child { background-color: #eeeeee; color: #333; font-weight: bold }
    table thead th { background-color: #eee; color: #000; }
    tr, th, td { border: 1px solid #ccc; border-width: 1px 0 0 1px; border-collapse: collapse;
    padding: 3px; font-family: monospace; font-size: 10px }</style>
    """
    s  = '<script type="text/Javascript">'
    s += 'var win = window.open("", "Title", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
    s += 'win.document.body.innerHTML = \'' + (df.to_html() + css).replace("\n",'\\') + '\';'
    s += '</script>'
    return(HTML(s+css))


# In[22]:


View(dataframe)

