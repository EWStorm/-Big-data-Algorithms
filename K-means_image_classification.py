#!/usr/bin/env python
# coding: utf-8

# In[85]:


import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k
import warnings
warnings.filterwarnings('ignore')

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting library
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

from keras.models import Sequential
from keras.layers import Dense , Activation, Dropout
from keras.optimizers import Adam ,RMSprop


import time
import datetime as dt



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output


# #https://www.kaggle.com/code/prashant111/comprehensive-guide-to-ann-with-keras/notebook
#https://medium.com/@joel_34096/k-means-clustering-for-image-classification-a648f28bdc47
#https://github.com/sharmaroshan/MNIST-Using-K-means/blob/master/KMeans%20Clustering%20for%20Imagery%20Analysis%20(Jupyter%20Notebook).ipynb

#check other metrics for accurasy score


# #Create the train data and test data

# In[86]:


#Test data: Used for testing the model that how our model has been trained. 
#Train data: Used to train our model.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# count the number of unique train labels
unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))


# count the number of unique test labels
unique, counts = np.unique(y_test, return_counts=True)
print("\nTest labels: ", dict(zip(unique, counts)))


# #data visualisation

# In[87]:


# sample 49 mnist digits from train dataset
indexes = np.random.randint(0, x_train.shape[0], size=49)
images = x_train[indexes]
labels = y_train[indexes]


# plot the 49 mnist digits
plt.figure(figsize=(7,7))
for i in range(len(indexes)):
    plt.subplot(7, 7, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
plt.show()
plt.savefig("mnist-samples.png")
plt.close('all')


# #Normalisation

# In[88]:


# img_rows and img_cols are used as the image dimensions. In mnist dataset, it is 28 and 28
#check the data format i.e. ‘channels_first’ or ‘channels_last’
print('Shape of x_train: ', x_train.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of x_test: ', x_test.shape)
print('Shape of y_test: ', y_test.shape)

print(x_train.dtype)
#3 dimentions 


# #Data normalization in Keras 

# In[89]:


#The MNIST images of 28×28 pixels are represented as an array of numbers whose values range from [0, 255] of type uint8.
#It is usual to scale the input values of neural networks to certain ranges.
#In this example, the input values should be scaled to values of type float32 within the interval [0, 1]. 


# In[90]:


# Checking the minimum and maximum values of x_train
print(x_train.min())
print(x_train.max())
#


# In[91]:


#reshape -->to reshape in such a way that we have we can access every pixel of the image.
#The input data have to be converted from 3 dimensional format to 2 dimensional format to be fed into the K-Means Clustering algorithm. 
#Hence the input data has to be reshaped.

X_train = x_train.reshape(len(x_train),-1)
X_test = x_test.reshape(len(x_test),-1)

#X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
#X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')

print(X_train.shape)
print(X_test.shape)


# In[92]:


#each pixel has its unique color code . To perform Machine Learning, it is important to convert all 
#the values from 0 to 255 for every pixel to a range of values from 0 to 1. 
#The simplest way is to divide the value of every pixel by 255 to get the values in the range of 0 to 1.
X_train = X_train / 255
X_test = X_test / 255

#checking min and max values
print(X_train.min())
print(X_train.max())


# In[93]:


#another way of doing the same 

# scale the input values to type float32

#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')

# scale the input values within the interval [0, 1]

#x_train /= 255
#x_test /= 255

#To facilitate the entry of data into our neural network we must make a transformation of the tensor (image) from 2 dimensions 
#(2D) to a vector of 1 dimension (1D).
#That is, the matrix of 28×28 numbers can be represented by a vector (array) of 784 numbers (concatenating row by row), 
#which is the format that accepts as input a densely connected neural network.
#In Python, converting every image of the MNIST dataset to a vector with 784 components can be accomplished as follows:

#x_train = x_train.reshape(60000, 784)
#x_test = x_test.reshape(10000, 784)


# In[94]:


#visualisation

#plt.gray() # B/W Images
#plt.figure(figsize = (10,9)) # Adjusting figure size
# Displaying a grid of 3x3 images
#for i in range(9):
 #plt.subplot(3,3,i+1)
 #plt.imshow(X_train[i])


# In[95]:


#visualisation
# create figure with 3x3 subplots using matplotlib.pyplot
#fig, axs = plt.subplots(3, 3, figsize = (12, 12))
#plt.gray()

# loop through subplots and add mnist images
#for i, ax in enumerate(axs.flat):
 #   ax.matshow(x_train[i])
  #  ax.axis('off')
   # ax.set_title('Number {}'.format(y_train[i]))
    
# display the figure
#fig.show()


# In[96]:


# Printing examples in 'y_train'
for i in range(5):
  print(y_train[i])


# #Building the model

# In[97]:


#init: Initialization method of the centroids
#n_clusters: The number of clusters to form as well as the number of centroids to generate. 
#n_init: Number of time the k-means algorithm will be run with different centroid seeds. 


# In[98]:


#Mini Batch K-Means works similarly to the K-Means algorithm. 
#The difference is that in mini-batch k-means the most computationally costly step is conducted on only a 
#random sample of observations as opposed to all observations. This approach can significantly reduce the 
#time required for the algorithm to find convergence with only a small cost in quality.


# In[99]:


from sklearn.cluster import MiniBatchKMeans
total_clusters = len(np.unique(y_test)) #10
print(total_clusters)

# K-Means model
kmeans = MiniBatchKMeans(n_clusters = total_clusters)
# Fitting the model to training set
kmeans.fit(X_train)


# In[100]:


cluster_sizes = [len(labels[labels==x]) for x in range(10)]
print(cluster_sizes)


# In[101]:


Labels=kmeans.labels_
print(Labels)
#But these are not real label of each image, since the output of the kmeans.labels_ is just group id for clustering.


# In[102]:


#To match it with real label, we can tackle the follow things:

#Combine each images in the same group
#Check Frequency distribution of actual labels (using np.bincount)
#Find the Maximum frequent label (through np.argmax), and set the label.


# In[103]:


#But the kmeans.labels_ only shows the cluster to which the image belongs to. 
#It doesn’t determine the number displayed in image.
#we need a fucntion-->that can Associate most probable label with each cluster in KMeans model
#and return dictionary of clusters assigned to each label.


# In[129]:


#function 
# from: https://github.com/sharmaroshan/MNIST-Using-K-means/blob/master/KMeans%20Clustering%20for%20Imagery%20Analysis%20(Jupyter%20Notebook).ipynb
#https://goodboychan.github.io/python/machine_learning/natural_language_processing/vision/2020/10/26/01-K-Means-Clustering-for-Imagery-Analysis.html#Normalization

def given_cluster_labels(kmeans, actual_labels):
    
    given_labels = {} #return list 

    # Loop through the clusters
    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # find most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the given_labels dictionary
        if np.argmax(counts) in given_labels:
            
            # append the new number to the existing array at this slot
            given_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            given_labels[np.argmax(counts)] = [i]

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))
        
    return given_labels  


# In[130]:


#Determines label for each array, depending on the cluster it has been assigned to.
 #   returns: predicted labels for each array
def given_data_labels(X_labels, cluster_labels):
    
    # empty array 
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels


# In[131]:


# test the given_cluster_labels() and given_data_labels() functions
cluster_labels = given_cluster_labels(kmeans, y_train)
X_clusters = kmeans.predict(X_train)
predicted_labels = given_data_labels(X_clusters, cluster_labels)

# Comparing Predicted values and Actual values
print(predicted_labels[:20].astype('int'))
print(y_train[:20])


# In[132]:


#some predicted label is mismatched


# In[133]:


from sklearn.metrics import accuracy_score
print(accuracy_score(predicted_labels,y_train))


# #Optimizing the Algorithm

# In[134]:


#homogeneity_score- Homogeneity metric of a cluster labeling given a ground truth.
#A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.

#inertia 

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html
#https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html?highlight=inertia


# In[135]:


from sklearn.metrics import homogeneity_score

def calc_metrics(estimator, data, labels):
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    # Inertia
    inertia = estimator.inertia_
    print("Inertia: {}".format(inertia))
    # Homogeneity Score
    homogeneity = homogeneity_score(labels, estimator.labels_)
    print("Homogeneity score: {}".format(homogeneity))
    return inertia, homogeneity


# In[146]:


from sklearn.metrics import accuracy_score

clusters = [10, 16, 36, 64, 144, 256]
iner_list = []
homo_list = []
acc_list = []

for n_clusters in clusters:
    estimator = MiniBatchKMeans(n_clusters=n_clusters)
    estimator.fit(X_train)
    
    inertia, homo = calc_metrics(estimator, X_train, y_train)
    iner_list.append(inertia)
    homo_list.append(homo)
    
    # Determine predicted labels
    cluster_labels = given_cluster_labels(estimator, y_train)
    prediction = given_data_labels(estimator.labels_, cluster_labels)
    
    acc = accuracy_score(y_train, prediction)
    acc_list.append(acc)
    print('Accuracy: {}\n'.format(acc))


# In[154]:


import matplotlib.pyplot as plt
plt.plot(iner_list,homo_list, acc_list)
plt.ylabel('some numbers')
plt.show()


# In[147]:


fig, ax = plt.subplots(1, 2, figsize=(16, 10))
ax[0].plot(clusters, iner_list, label='inertia', marker='o')
ax[1].plot(clusters, homo_list, label='homogeneity', marker='o')
ax[1].plot(clusters, acc_list, label='accuracy', marker='^')
ax[0].legend(loc='best')
ax[1].legend(loc='best')
ax[0].grid('on')
ax[1].grid('on')
ax[0].set_title('Inertia of each clusters')
ax[1].set_title('Homogeneity and Accuracy of each clusters')
plt.show()
#comments:
# when the K value is increased, the accuracy and homogeneity is also increased, The inertia score decreases. 
#the homogeneity score--> increases the clusters become more differentiable and number of data points having a single class label is high
##The accuracy is highest for ‘number of clusters’ = 256. 


# In[150]:


#check the performance on test dataset, K=256.
## test kmeans algorithm on testing dataset
# Initialize the K-Means model
kmeans = MiniBatchKMeans(n_clusters = 256)

# Fitting the model to testing set
kmeans.fit(X_test)


# In[151]:


cluster_labels = given_cluster_labels(kmeans, y_test)

test_clusters = kmeans.predict(X_test)
prediction = given_data_labels(kmeans.predict(X_test), cluster_labels)
print('Accuracy: {}'.format(accuracy_score(y_test, prediction)))
print('Number of Clusters: {}'.format(estimator.n_clusters))


# In[169]:


#visualising 

kmeans = MiniBatchKMeans(n_clusters = 36)
kmeans.fit(X_test)

# record centroid values
centroids = kmeans.cluster_centers_

# reshape centroids into images
images = centroids.reshape(36, 28, 28)
images *= 255
images = images.astype(np.uint8)

# determine cluster labels
cluster_labels = given_cluster_labels(kmeans, y_test)
prediction = given_data_labels(kmeans.predict(X_test), cluster_labels)

# create figure with subplots using matplotlib.pyplot
fig, axs = plt.subplots(6, 6, figsize = (30, 30))
plt.gray()

# loop through subplots and add centroid images
for i, ax in enumerate(axs.flat):
    
    # determine inferred label using cluster_labels dictionary
    for key, value in cluster_labels.items():        
        if i in value:
            ax.set_title('Given Label: {}'.format(key), color='blue')
    
    # add image to subplot
    ax.matshow(images[i])
    ax.axis('off')
    
# display the figure
plt.show()


# #visualisation of centroids

# In[74]:


kmeans = MiniBatchKMeans(n_clusters = 36)
kmeans.fit(X_test)


# In[75]:


centroids = kmeans.cluster_centers_


# In[76]:


centroids.shape
#256 cluster centroids, and each cluster centroid has 784 features.


# In[77]:


#reshaping centroids from 2D format to 3D for visualisation. 
#back into a 28 by 28 pixel image and plot 
centroids = centroids.reshape(36,28,28)
centroids = centroids * 255


# In[78]:


centroids.shape


# In[79]:


plt.figure(figsize = (10,10))
for i in range(36):
 plt.subplot(6,6,i+1)
 plt.imshow(centroids[i])


# In[ ]:


#conclusion: 
#made MNIST classifier with almost 90% accuracy 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




