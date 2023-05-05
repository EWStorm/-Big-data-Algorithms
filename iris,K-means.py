#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import os



import warnings
warnings.filterwarnings('ignore') #to supress warnings 


# In[3]:


#loading iris dataset
iris= datasets.load_iris()
#Create futures 
X=iris.data # rows 

#Create label 
y=iris.target 

#print(X)
#print(y)

#view row number 1-->[0], 2-->[1]...
print(X[0])
X.shape #150 rows, 4 colums 


# In[4]:


iris


# In[ ]:





# In[126]:


dir(iris)


# In[127]:


feature_names = iris.feature_names
target_names = iris.target_names
print("Feature names:", feature_names) 
print("Target names:", target_names)
print("\nFirst 10 rows of X:\n", X[:10])


# In[128]:


data_module=iris.data_module
frame=iris.frame
filename=iris.filename 
print("data module:   ", data_module)
print(" frame:  ", frame)
print(" filename:  ", filename)


# #Importer KMeans fra sklearn.cluster Instantier KMeans med tre clustre (der er tre typer Iris)) Træn Kmeans med iris datasættet. Prøv at kvalificer resultatet (kan algoritmen genkende ireis typerne korrekt)
# 
# #K-means ckustering- is an unsupervised algorithm that is used for predicting grouping from within the unlabeled dataset.
# 
# #STEP1: number of clusters (K=3)
# 
# #STEP2: randomly select 3 distinc data points
# 
# #STEP3: measure the distance between the 1st point and the 3 initial clusters
# 
# #STEP4: assign the 1 point to the nearest cluster.
# 
# #STEP5: calculate the mean of each cluster.
# 

# #convert arrays to a pandas DataFrame for ease of use

# In[129]:


X = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])


# In[130]:


X


# In[131]:


y


# In[132]:


target_names = iris.target_names
print("Target names:", target_names)


# #scatter plot 

# In[133]:


# plot figure of size 12 units wide & 3 units tall
plt.figure(figsize=(12,3))
# Create an array of three colours, one for each species.
colors = np.array(['red', 'green', 'blue'])

#Draw a Scatter plot for Sepal Length vs Sepal Width
#nrows=1, ncols=2, plot_number=1
# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot
plt.subplot(1, 2, 1)

# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
plt.scatter(X['Sepal Length'], X['Sepal Width'], c=colors[y['Target']], s=40)
plt.title('Sepal Length vs Sepal Width')

plt.subplot(1,2,2)
plt.scatter(X['Petal Length'], X['Petal Width'], c= colors[y.Target], s=40)
plt.title('Petal Length vs Petal Width')


# #use the KMeans algorithm to see if it can create the clusters automatically

# In[134]:


import warnings
warnings.filterwarnings('ignore') #to supress warnings 

# create a model object with 3 clusters
model = KMeans(n_clusters=3)
model.fit(X) #model.fit() function runs the algo on the data and creates the clusters. 
#Each sample in the dataset is then assigned a cluster id (0, 1, 2, etc).

#Start with a plot figure of size 12 units wide & 3 units tall
plt.figure(figsize=(12,3))

# Create an array of three colours, one for each species.
colors = np.array(['red', 'green', 'blue'])

# The fudge to reorder the cluster ids.
predictedY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)

# Plot the classifications that we saw earlier between Petal Length and Petal Width
plt.subplot(1, 2, 1)
plt.scatter(X['Petal Length'], X['Petal Width'], c=colors[y['Target']], s=40)
plt.title('Before classification')
 
# Plot the classifications according to the model
plt.subplot(1, 2, 2)
plt.scatter(X['Petal Length'], X['Petal Width'], c=colors[predictedY], s=40)
plt.title("Model's classification") 


# #from ytube video 

# In[135]:


model = KMeans(n_clusters=3, init="k-means++")
model.fit(X) #model.fit() function runs the algo on the data and creates the clusters. 
#Each sample in the dataset is then assigned a cluster id (0, 1, 2, etc).


# In[136]:


model.cluster_centers_  # 3 clusters, 


# In[137]:


model.labels_ # stores an array of f the cluster ids


# In[138]:


X['clusters']=model.labels_ #add a new column to the original data-->iris.data, matrix 'X'. 


# In[139]:


X


# In[140]:


X['clusters'].value_counts() #values in each cluster


# # trying different functions from youtube video on K-means

# In[141]:


df=pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])


# In[142]:


df


# In[143]:


df.columns


# In[ ]:




