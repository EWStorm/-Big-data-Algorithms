#!/usr/bin/env python
# coding: utf-8

# In[258]:


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
import plotly
import plotly.graph_objs as go


import warnings
warnings.filterwarnings('ignore') #to supress warnings

import matplotlib.pyplot as plt


# #  Load the data into a pandas dataframe
# • Initialise K-means with 6 clusters
# • Run the K-means ‘fit’ on the acceleration data
# • The three acceleration columns can be selected by using a list of columns as
# index in the pandas dataframe

# In[140]:


os.getcwd()


# In[141]:


os.path.abspath('gitHub5')


# In[142]:


root= 'C:\\Users\\Ekaterina\\desktop\\gitHub5.csv'
r_data=pd.read_csv(root,index_col=0) # dada frame object (pandas)


# In[143]:



len(r_data.columns)


# In[144]:


r_data.columns


# In[145]:


df = pd.DataFrame(r_data)


# In[146]:


df.shape


# In[147]:


df.head()


# In[171]:


df.info()


# In[148]:


df.columns


# In[149]:


# 
acc = df[['accelerometerAccelerationX','accelerometerAccelerationY','accelerometerAccelerationZ']]


# In[150]:


#renaming the columns 
acc.columns = ['Acc_X', 'Acc_Y', 'Acc_Z']


# In[151]:


# creating a model with 6 clusters 

kmeans = KMeans(n_clusters=6, random_state=0) #Model 
kmeans = kmeans.fit(acc)
 


# In[152]:


#predict the labels of clusters.
label = kmeans.fit_predict(acc)
 
print(label) #clusters labels 


# In[153]:



kmeans.cluster_centers_


# In[154]:


kmeans.labels_ # stores an array of f the cluster ids


# In[155]:


acc.plot()
plt.show()


# In[156]:


acc['clusters']=kmeans.labels_ #add a new column to the original data-->iris.data, matrix 'X'. 
acc['clusters'].value_counts() #values in each cluster


# In[157]:


#from: https://towardsdatascience.com/visualizing-clusters-with-pythons-matplolib-35ae03d87489


# In[158]:


acc['cluster'] = kmeans.fit_predict(acc[['Acc_X','Acc_Y','Acc_Z']])


# In[159]:


centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]


# In[160]:


## add to data frame 
acc['cen_x'] = acc.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2], 3:cen_x[3], 4:cen_x[4], 5:cen_x[5]})
acc['cen_y'] = acc.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2], 3:cen_y[3], 4:cen_y[4], 5:cen_y[5]})


# In[161]:


acc


# In[162]:


#define colors for each cluster, add to df


# In[163]:


# define and map colors
colors = ['#DF2020', '#81DF20', '#2095DF', '#bf00bf','#00bfbf', '#000000']
acc['c'] = acc.cluster.map({0:colors[0], 1:colors[1], 2:colors[2],3:colors[3],4:colors[4],5:colors[5] })


# In[164]:


plt.scatter(acc.Acc_X, acc.Acc_Y, c=acc.c, s=40, alpha = 0.6)
plt.title('Acc X vs Acc Y')


# #multi dimentional 

# In[165]:


from mpl_toolkits.mplot3d import Axes3D


#add the third colum to plot --> acceleration Z
fig = plt.figure(figsize=(26,6))
ax = fig.add_subplot(131, projection='3d')
ax.scatter(acc.Acc_X, acc.Acc_Y, acc.Acc_Z, c=acc.c, s=40)
ax.set_xlabel('Acc on X')
ax.set_ylabel('Acc on Y')
ax.set_zlabel('Acc on Z')
plt.title('Acc X vs Acc Y vs Acc Z')
plt.show()


# In[166]:


from matplotlib.lines import Line2D

fig, ax = plt.subplots(1, figsize=(8,8))
# plot data
plt.scatter(acc.Acc_X, acc.Acc_Y, c=acc.c, s=50, alpha = 0.6)
#plt.title('Acc X vs Acc Y') # create a list of legend elemntes
## markers / records
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), 
               markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]# plot legend
plt.legend(handles=legend_elements, loc='upper right')# title and labels
plt.title('Acceleration X vs Acceleration Y', loc='left', fontsize=22)
plt.xlabel('Acc X')
plt.ylabel('Acc Y')


# #centroids, Mean values. 

# In[167]:



# plot centroids
plt.scatter(cen_x, cen_y, marker='*', c=colors, s=70)

## plot centroids
plt.scatter(cen_x, cen_y, marker='^', c=colors, s=70)


# #Lines
# 
# In k-means, since we’re working with distances, connecting the points to their respective centroids can help us visualize what the algorithm is actually doing.

# In[174]:


from matplotlib.lines import Line2D


fig, ax = plt.subplots(1, figsize=(12,12))
# plot data
plt.scatter(acc.Acc_X, acc.Acc_Y, c=acc.c, s=50, alpha = 0.6)
#plt.title('Acc X vs Acc Y') # create a list of legend elemntes

# plot centroids
plt.scatter(cen_x, cen_y, marker='^', c=colors, s=70)
plt.scatter(cen_x, cen_y, marker='*', c=colors, s=70)

# plot Acc_X mean
#plt.plot([acc.Acc_X.mean()]*2, color='black', lw=0.5, linestyle='--')
#plt.xlim(0,5)

# plot Acc_Y mean
#plt.plot ([0,2],[acc.Acc_Y.mean()]*2, color='black', lw=0.5, linestyle='--')
#plt.ylim(0,5)


# plot lines
for idx, val in acc.iterrows():
    x = [val.Acc_X, val.cen_x,]
    y = [val.Acc_Y, val.cen_y]
    plt.plot(x, y, c=val.c, alpha=0.2)

## markers / records
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), 
               markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]
# plot legend
plt.legend(handles=legend_elements, loc='upper right')
# title and labels
plt.title('Acceleration X vs Acceleration Y', loc='left', fontsize=22)
plt.xlabel('Acc X')
plt.ylabel('Acc Y')


#stranno vigljadit :-((


# #The convex hull is the smallest set of connections between our data points to form a polygon that encloses all the points, and there are ways to find the convex hull systematically — That is to say, we can use Sklearn to get the contour of our dataset.

# In[184]:


from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
#from scipy import interpolate

fig, ax = plt.subplots(1, figsize=(12,12))
# plot data
plt.scatter(acc.Acc_X, acc.Acc_Y, c=acc.c, s=50, alpha = 0.6)
# plot centroids
plt.scatter(cen_x, cen_y, marker='^', c=colors, s=70)
plt.scatter(cen_x, cen_y, marker='*', c=colors, s=70)

# draw enclosure
for i in acc.cluster.unique():
    points = acc[acc.cluster == i][['Acc_X', 'Acc_Y']].values
    # get convex hull
    hull = ConvexHull(points)
    # get x and y coordinates
    # repeat last point to close the polygon
    x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])
    # plot shape
    plt.fill(x_hull, y_hull, alpha=0.3, c=colors[i])
    
    # interpolate
    #dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
    #dist_along = np.concatenate(([0], dist.cumsum()))
    #spline, u = interpolate.splprep([x_hull, y_hull], 
                                 #   u=dist_along, s=0, per=1)
    #interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    #interp_x, interp_y = interpolate.splev(interp_d, spline)
    # plot shape
    #plt.fill(interp_x, interp_y, '--', c=colors[i], alpha=0.2)
    
#plt.xlim(0,200)
#plt.ylim(0,200)


# # extraction location data and doing kmeans

# In[201]:


r_data


# In[205]:


dff = pd.DataFrame(r_data)


# In[207]:


dff.columns


# In[212]:


location=dff[['locationLongitude', 'locationAltitude',  'locationLatitude']]


# In[213]:


location.columns=['Longitude', 'Altitude', 'Latitude']


# In[215]:


location


# In[218]:


# creating a model with 6 clusters 

kmeans = KMeans(n_clusters=6, random_state=0) #Model 
kmeans = kmeans.fit(location)

#predict the labels of clusters.
label = kmeans.fit_predict(location)
 
print(label) #clusters labels 


# In[219]:


kmeans.cluster_centers_


# In[227]:


#cluster size 

cluster_sizes = [len(label[label==x]) for x in range(6)]
cluster_sizes


# #Plotting Label 0 K-Means Clusters

# In[279]:


filtered_label0 = location[label == 0]
filtered_label_1 = location[label == 1]
filtered_label_2 = location[label == 2]
filtered_label_3 = location[label == 3]
filtered_label_4 = location[label == 4]
filtered_label_5 = location[label == 5]

# convert to arrays 
lab0=filtered_label0.to_numpy()
lab1=filtered_label_1.to_numpy()
lab2=filtered_label_2.to_numpy()
lab3=filtered_label_3.to_numpy()
lab4=filtered_label_4.to_numpy()
lab5=filtered_label_5.to_numpy()


# In[282]:


lab5


# In[288]:


plt.scatter(lab0[:,0] , lab0[:,1] , color = 'red') #plot first cluster 
plt.scatter(lab1[:,0] , lab1[:,1] , color = 'blue') #plot first cluster 
plt.scatter(lab2[:,0] , lab2[:,1] , color = 'green') #plot first cluster
plt.scatter(lab3[:,0] , lab3[:,1] , color = 'black') #plot first cluster
plt.scatter(lab4[:,0] , lab4[:,1] , color = 'brown') #plot first cluster
plt.scatter(lab5[:,0] , lab5[:,1] , color = 'yellow') #plot first cluster


# In[290]:


# another way for the same plot 
#Getting unique labels
 
#u_labels = np.unique(label)
 
#plotting the results:
 
#for i in u_labels:
 #   plt.scatter(location[label == i , 0] , location[label == i , 1] , label = i)
#plt.legend()
#plt.show()


# In[297]:


plt.scatter(centroids[:,0] , centroids[:,1] , s = 20, color = 'black')
plt.legend()
plt.show()


# In[233]:


r_data['cluster'] =label 

cluster_speed = r_data[['locationSpeed','cluster']]
#cluster_speed



# In[234]:


cluster_speed.plot(y='locationSpeed')


# In[237]:




for i in range(6):
    cluster_speed[f'cluster{i}'] = cluster_speed['locationSpeed'][cluster_speed['cluster']==i]
  


# In[238]:


cluster_speed.plot(y=['cluster0','cluster1','cluster2','cluster3','cluster4','cluster5'])


# In[ ]:





# In[266]:


location['Longitude'].nunique()


# In[267]:


loc_values = list(set(location['Longitude']))
#loc_values 


# In[ ]:




