#!/usr/bin/env python
# coding: utf-8

# In[364]:


from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import r2_score, explained_variance_score, confusion_matrix, accuracy_score, classification_report, log_loss
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os ## to display the current working dirrectory 
import datetime as dt 



import warnings
warnings.filterwarnings('ignore') #to supress warnings


# In[365]:


#load the raw data 
os.getcwd()


# In[366]:


os.listdir(os.getcwd())


# In[367]:


#raw_data=pd.read_csv('C:\Users\Ekaterina\Desktop\Python\file_bdata1.csv')


# In[368]:


os.path.abspath('bigData_4.csv')


# In[369]:


os.path.exists('bigData_4.csv') #checks whether a file or directory exists


# #loading raw data

# In[370]:


ls \Users\Ekaterina\desktop\     #check out the path 


# In[375]:


root= 'C:\\Users\\Ekaterina\\desktop\\bigData_4.csv'
raw_data=pd.read_csv(root,encoding='latin1',skiprows=20,sep=';') # dada frame object (pandas)

#raw_data=pd.read_csv('bigData_4.csv')


# # downloading the names of 20 columns of the data 
# 
# 

# In[376]:


with open(root,'r',encoding='latin1') as f:
    dat = f.read()
    print(dat[:1000])


# In[404]:


import codecs

def read_energi_data(f):                              #function for data reading and sorting columns 
    with codecs.open(root,encoding='8859') as f:
        col = []
        l=f.readline().strip()
        while l:
            col.append(' '.join(l.split()[1:]))
            l=f.readline().strip()
        
        dat = pd.read_csv(f,sep=';', skipinitialspace=True,
                          lineterminator ='\n',
                          infer_datetime_format=True)
        dat = dat.drop(dat.columns[-1],1)
        
        dat.columns = [dat.columns[0]] + col
        dat = dat.set_index(dat.columns[0])
    return dat


# In[405]:



df = read_energi_data(root)


# In[406]:


df


# In[407]:


df.columns


# In[408]:


import matplotlib.pyplot as plt
df.plot()
plt.show()


# In[409]:


df[['CO2 udledning', 'Solceller DK2']].plot()


# In[410]:


df[['CO2 udledning', 'Solceller DK2','Vindhastighed i Malling',]].plot()


# #another way of doing the same 

# In[411]:


raw_data
raw_data.shape


# In[378]:


#first 5 raws 
raw_data.head()


# #data preprocessing

# In[379]:


#Exploratory Data Analysis & Data Cleaning
## Checking for null values
#raw_data.isnull().sum()


#  #plot the columns of the dataframe (a plot for each column, and in a few plots).

# #convert #convert arrays to a pandas DataFrame for ease of use
# 

# In[380]:


df = pd.DataFrame(raw_data)
df


# #plot with pandas

# In[381]:


import matplotlib.pyplot as plt
df.plot()
plt.show()


# #had to input names in lst manually here. above, there is a more elegant way of duing this in [376]. 
# 

# In[382]:


lst=['Date&time','Centrale kraftværker DK1','Centrale kraftværker DK2', 'Decentrale kraftværker DK1',
      'Decentrale kraftværker DK2', 'Vindmøller DK1', 'Vindmøller DK2', 'Udveksling Jylland-Norge', 
      'Udveksling Jylland-Sverige', 'Udveksling Jylland-Tyskland', 'Udveksling Sjælland-Sverige', 
      'Udveksling Sjælland-Tyskland','Udveksling Bornholm-Sverige', 'Udveksling Fyn-Sjaelland','Temperatur i Malling',
'Vindhastighed i Malling','CO2 udledning', 'Havmøller DK', 'Landmøller DK', 'Solceller DK1','Solceller DK2','']


# In[383]:


df.columns =[lst]


# In[384]:


df


# In[385]:


df.plot()
plt.show()


# In[386]:


df[['CO2 udledning', 'Solceller DK2']].plot()


# In[387]:


df[['CO2 udledning', 'Vindmøller DK2','Vindmøller DK1']].plot()


# In[388]:


df[['CO2 udledning', 'Vindmøller DK2','Vindmøller DK1', 'Centrale kraftværker DK2']].plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




