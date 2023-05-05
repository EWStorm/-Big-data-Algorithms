#!/usr/bin/env python
# coding: utf-8

# In[30]:


# broadly used of unsupervised algorithms, PCA. #PCA is fundamentally a 
# dimensionality reduction algorithm, but it can also be useful as a tool for visualization, 
# for noise filtering, for feature extraction etc. 


# In[58]:


from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns
from sklearn.metrics import r2_score, explained_variance_score, confusion_matrix, accuracy_score, classification_report, log_loss
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os ## to display the current working dirrectory 
import datetime as dt 

from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore') #to supress warnings

import matplotlib.pyplot as plt
sns.set()


# #Load the data into a pandas dataframe

# In[59]:


os.getcwd()


# In[60]:


os.path.abspath('elProd')


# In[61]:


root= 'C:\\Users\\Ekaterina\\desktop\\elProd.csv'
raw_data=pd.read_csv(root,encoding='latin1',skiprows=20,sep=';') # dada frame object (pandas)

#raw_data=pd.read_csv('elProd')


# In[62]:


raw_data


# In[63]:


#names to columns 
raw_data.columns


# #ownloading the names of 20 columns of the data

# In[64]:


with open(root,'r',encoding='latin1') as f:
    dat = f.read()
    print(dat[:1000])


# In[65]:


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


# In[66]:


df = read_energi_data(root)


# In[67]:


df


# In[68]:


#CO2 column og smth
CO2=df[['CO2 udledning', 'Havmøller DK', 'Udveksling Bornholm-Sverige', 'Udveksling Jylland-Tyskland']]


# In[69]:


CO2


# In[70]:


# quick check of the data 
CO2.plot() #only 2 variables 
plt.show()
df.plot() #all dataframe 


# In[71]:


#plot only the columns of the data (df) with the data from Vindmøller
df['Vindmøller DK1'].plot()


# In[72]:


df.plot.scatter(x='Vindmøller DK2', y='Vindmøller DK1', alpha=0.5)

plt.show()


# In[73]:


#convert dataframe to array (numphy) df.to_numpy()
array_Data=CO2.to_numpy()


# In[74]:


CO2.plot.scatter(x='Havmøller DK', y='CO2 udledning', c='blue', s=40) #plot data frame 


# In[75]:


#plot the array 
plt.scatter(array_Data[:, 0], array_Data[:, 1])
plt.axis('equal');


# #PCA. from https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

# #relationship between the x and y values. 
# #in PCA In principal component analysis, this relationship is quantified by finding a list of the principal axes in the data, and using those axes to describe the dataset. 

# In[76]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(array_Data)


# In[77]:


#The fit learns some quantities from the data, most importantly the "components" and "explained variance":


# In[78]:


print(pca.components_)


# In[79]:


print(pca.explained_variance_)


# # PCA as dimensionality reduction

# In[81]:


#sing PCA for dimensionality reduction involves zeroing out one or more of the smallest principal components, 
#resulting in a lower-dimensional projection of the data that preserves the maximal data variance


# In[82]:


#Here is an example of using PCA as a dimensionality reduction transform:


# In[83]:


pca = PCA(n_components=1)
pca.fit(array_Data)
X_pca = pca.transform(array_Data)
print("original shape:   ", array_Data.shape)
print("transformed shape:", X_pca.shape)


# In[84]:


#The transformed data has been reduced to a single dimension. To understand the effect of this dimensionality reduction, 
#we can perform the inverse transform of this reduced data 
#and plot it along with the original data:


# In[85]:


X_new = pca.inverse_transform(X_pca)
plt.scatter(array_Data[:, 0], array_Data[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');


# #above -->The light points are the original data, while the dark points are the projected version. This makes clear what a PCA dimensionality reduction means: the information along the least important principal axis or axes is removed, leaving only the component(s) of the data with the highest variance. The fraction of variance that is cut out (proportional to the spread of points about the line formed in this figure) is roughly a measure of how much "information" is discarded in this reduction of dimensionality.

# #from: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

# In[86]:


#the same dataFrame df and reduced df -->CO2
#PCA, or Principal component analysis, is the main linear algorithm for dimension reduction often 
#used in unsupervised learning. This algorithm identifies and discards features that are less useful to make a valid 
#approximation on a dataset.

#1. scaling the data 


# In[87]:


from sklearn.preprocessing import StandardScaler
features = ['CO2 udledning', 'Havmøller DK', 'Udveksling Bornholm-Sverige','Udveksling Jylland-Tyskland']

# Separating out the features
x = CO2.loc[:, features].values

# Separating out the target
#y = df.loc[:,['target']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)


# In[88]:


#The array x (visualized by a pandas dataframe) before and after standardization
x #CO2 scaled to x  array (standartization)


# In[89]:


x.shape


# In[90]:


#reducing dimentions to 1 
pca = PCA(n_components=1)
pca.fit(x)
X_pca = pca.transform(x)
print("original shape:   ", x.shape)
print("transformed shape:", X_pca.shape)


# In[91]:


principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1'])


# In[92]:


print(principalDf)


# In[ ]:





# In[ ]:





# In[ ]:




