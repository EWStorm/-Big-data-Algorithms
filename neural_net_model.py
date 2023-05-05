#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Neural networks are a series of algorithms that identify underlying relationships in a set of data. 
#These algorithms are heavily based on the way a human brain operates. These networks can adapt to changing input and 
#generate the best result without the requirement to redesign the output criteria. In a way, these neural networks are 
#similar to the systems of biological neurons (1).


# In[ ]:


#The purpose: to train a Neural Network classification on MNIST data set: MLP with Keras.
#training an MLP with sparse cross-entropy loss function. 
#Multi-Class Classification Problem


# In[ ]:


pip install pydot


# In[ ]:



pip install keras 


# In[ ]:


pip install tensorflow 


# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k
from keras.utils import to_categorical, plot_model

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential 


# In[2]:


#importing dataset
mnist = tf.keras.datasets.mnist

#dataset is already split into a training set and a test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

#every image is represented as a 28×28 array. the pixel intensities are represented as integers (from
#0 to 255) rather than floats (from 0.0 to 255.0) like in Scikit-Learn. Here is the shape and data type of the
#training set:
print(x_train.shape)
print(x_train.dtype)

#the shape of single image 
x_train[0].shape


# In[3]:


#visualisation of random digit 
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit=x_train[1]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


# In[4]:


#looks like 0, and check the label:
y_train[1]


# In[5]:


#detailed visualisation

def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
visualize_input(x_train[1], ax)

#(6)


# In[6]:


# creating a validation dataset. 
#we must scale the input features. For simplicity, we just
#scale the pixel intensities down to the 0-1 range by dividing them by 255.0 (this also
#converts them to floats(4)). 
X_valid, X_train = x_train[:5000] / 255.0, x_train[5000:] / 255.0
y_valid, y_train = y_train[:5000], y_train[5000:]


# In[7]:


# View number of dimensions of tensor
print(X_train.ndim)

# View the dimension of tensor
print(X_train.shape)
print(X_valid.shape)


# In[8]:


#each pixel has its unique color code . To perform Machine Learning, it is important to convert all 
#the values from 0 to 255 for every pixel to a range of values from 0 to 1. 
#The simplest way is to divide the value of every pixel by 255 to get the values in the range of 0 to 1.
#X_train = X_train / 255
#X_test = X_test / 255


# In[9]:


#Creating the Model Using the Sequential API

##The first line creates a model. This is the simplest kind of Keras
#Sequential model, for neural networks that are just composed of a single stack of layers, connected
#sequentially (4).
model = keras.models.Sequential() 

# build the first layer and add it to the model. flatten role is to convert each input image into a 1D array
model.add(keras.layers.Flatten(input_shape=[28, 28]))

#add a Dense hidden layer with 300 neurons. use the ReLU activation function
model.add(keras.layers.Dense(300, activation="relu"))

#add a second Dense hidden layer with 100 neurons, also using the ReLU activation function.
model.add(keras.layers.Dense(100, activation="relu"))

#add a Dense output layer with 10 neurons (one per class), using the softmax activation function 
#(because the classes are exclusive).
model.add(keras.layers.Dense(10, activation="softmax"))

#arg max,” mathematical function returns the index in the list that contains the largest value.
#“soft max,” mathematical function can be thought to be a probabilistic or “softer” version of the argmax function.
#returns a list of probabilities. https://machinelearningmastery.com/softmax-activation-function-with-python/
#The softmax function is used as the activation function in the output layer of neural network models that predict 
#a multinomial probability distribution.


#from the file EX_ANN

#import tensorflow as tf
#from tensorflow.keras.layers import Dense,Flatten, Dropout, Activation, Conv2D, MaxPooling2D
#from tensorflow.keras.models import Sequential 

#model = Sequential()
#model.add(Dense(32, input_dim = 28 * 28, activation= 'relu'))
#model.add(Dense(64, activation = 'relu')) #Rectified Linear Unit activation function , f(x)=max(0,x).
#model.add(Dense(10, activation = 'softmax'))


# In[10]:


#displays all the model’s layers
model.summary()


# In[11]:


model.layers


# In[12]:


#model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#compiling the model 
model.compile(loss="sparse_categorical_crossentropy",
optimizer="sgd", #we will train the model using Stochastic Gradient Descent.Keras will perform the
#backpropagation algorithm
metrics=["accuracy"])

#we use the "sparse_categorical_crossentropy" loss because we have sparse labels (i.e., for each instance there is just a target
#class index, from 0 to 9 in this case), and the classes are exclusive (4,5).
# this loss function is Used for multi-class classification model where the output label is assigned integer 
#value (0, 1, 2, 3…). This loss function is mathematically same as the categorical_crossentropy. 
#It just has a different interface(5). 
#Sparse cross-entropy performsthe same cross-entropy calculation of error, without requiring that the target variable 
#be one hot encoded prior to training.


# In[13]:


#model.fit(X_train, y_train, epochs=10, batch_size=100)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), batch_size=100)


# In[14]:


#computes the loss and the accuracy of the model in the test set.
loss, acc = model.evaluate(x_test, y_test)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))


# In[15]:


train_acc = model.evaluate(X_train, y_train, verbose=1)
test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Train set results:  ',train_acc)
print('Test set result.   ',test_acc)


# In[16]:


from matplotlib import pyplot as plt

# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()


#A figure is also created showing two line plots, the top with the sparse cross-entropy loss over epochs for the train (blue) 
#and test (orange) dataset, and the bottom plot showing classification accuracy over epochs.
#In this case, the plot shows good convergence of the model over training with regard to loss and classification accuracy.


# In[17]:


#The neural network is trained. At each epoch during training, Keras displays
#the number of instances processed so far (along with a progress bar), the mean
#training time per sample, the loss and accuracy. we can see that the training loss
#went down, which is a good sign, and the validation accuracy reached 96% after 50
#epochs, not too far from the training accuracy, so there does not seem to be much
#overfitting going on

#Hence, after training the model we have achieved an accuracy of 99.88% for the training data set. 
#Now, it’s time to see how the model works in the test set and see whether we have achieved the required accuracy. 


# In[18]:


#plot the model 
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()


# In[19]:


#both the training and validation accuracy steadily increase during
#training, while the training and validation loss decrease
#validation curves are close to training curves---> not much overfitting 


# In[20]:


history.params


# In[21]:


#dictionary, containing the loss and extra metrics it
#measured at the end of each epoch on the training set and on the validation set

#history.history


# In[22]:


test_loss, test_acc = model.evaluate(x_test, y_test)


# In[23]:


#Predictions 
# we can use the model’s method predict() to make predictions on new instances 
#Since we don’t have actual new instances, we will just use the first 10 instances of
#the test set.
X_new = x_test[:10]
y_proba = model.predict(X_new)
y_proba.round(3)
#y_proba


# #tuning

# In[24]:


#The flexibility of neural networks is also one of their main drawbacks: there are many
#hyperparameters to work with. Not only can you use any imaginable network architecture,
#but even in a simple MLP you can change the number of layers, the number of
#neurons per layer, the type of activation function to use in each layer and etc. 
#GridSearchCV or RandomizedSearchCV


# In[25]:


pip install scikeras


# In[26]:


import warnings
warnings.filterwarnings('ignore')


# In[27]:


#to wrap our Keras models in objects that mimic regular Scikit-Learn regressors--> we need to 
#to create a function that will build and compile a Keras model, given a set of hyperparameters (4,9)
from scikeras.wrappers import KerasClassifier, KerasRegressor


# In[28]:


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    
    model = keras.models.Sequential()
    options = {"input_shape": input_shape}
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu", **options))
    options = {}
    model.add(keras.layers.Dense(1, **options))
    optimizer = keras.optimizers.SGD(learning_rate)
    
    model.compile(loss="mse", optimizer=optimizer)
    return model


# In[29]:


#create a KerasRegressor based on this build_model() function 
#The KerasRegressor object is a thin wrapper around the Keras model built using build_model()
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)


# In[30]:


#Since there are many hyperparameters, it is preferable to use a Randomized search
#rather than grid search.(4)


# In[31]:


#exploring the number of hidden layers, the number of neurons and the learning rate.

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
"n_hidden": [0, 1, 2, 3],
"n_neurons": np.arange(1, 100),
"learning_rate": reciprocal(3e-4, 3e-2),
}
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100,
validation_data=(X_valid, y_valid),
callbacks=[keras.callbacks.EarlyStopping(patience=10)])
# the RandomizedSearchCV uses K-fold validation, so it does not use X_valid and y_valid 


# In[33]:


#accessing the best parameters found, the best score, and the trained Keras model.
rnd_search_cv.best_params_


# In[34]:


rnd_search_cv.best_score_


# In[35]:


#save param-s. 
model = rnd_search_cv.best_estimator_.model


# In[36]:


model.save("my_keras_model.h5")


# In[ ]:


#model = keras.models.load_model("my_keras_model.h5")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#References:
#1) https://analyticsindiamag.com/top-5-neural-network-models-for-deep-learning-their-applications/
#2) https://towardsdatascience.com/why-data-should-be-normalized-before-training-a-neural-network-c626b7f66c7d
#3) https://machinelearningmastery.com/using-normalization-layers-to-improve-deep-learning-models/
#4) BOOK. page 294
#5) https://vitalflux.com/keras-categorical-cross-entropy-loss-function/
#6)https://www.kaggle.com/code/drouholi/mnist-mlp/notebook
#7) https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
#8) https://www.projectpro.io/article/exploring-mnist-dataset-using-pytorch-to-train-an-mlp/408
#9)https://mlfromscratch.com/gridsearch-keras-sklearn/#running-gridsearchcv

