#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Neural networks are a series of algorithms that identify underlying relationships in a set of data. 
#These algorithms are heavily based on the way a human brain operates. These networks can adapt to changing input and 
#generate the best result without the requirement to redesign the output criteria. In a way, these neural networks are 
#similar to the systems of biological neurons (1).


# In[2]:


#The purpose: to train a Neural Network classification on MNIST data set: MLP with Keras.
#training an MLP with sparse cross-entropy loss function. 
#Multi-Class Classification Problem


# In[1]:


pip install pandoc


# In[3]:


pip install pydot


# In[4]:



pip install keras 


# In[5]:


pip install tensorflow 


# In[6]:


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


# In[7]:


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


# In[8]:


#visualisation of random digit 
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit=x_train[1]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


# In[9]:


#looks like 0, and check the label:
y_train[1]


# In[10]:


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


# In[11]:


# creating a validation dataset. 
#we must scale the input features. For simplicity, we just
#scale the pixel intensities down to the 0-1 range by dividing them by 255.0 (this also
#converts them to floats(4)). 
X_valid, X_train = x_train[:5000] / 255.0, x_train[5000:] / 255.0
y_valid, y_train = y_train[:5000], y_train[5000:]


# In[12]:


# View number of dimensions of tensor
print(X_train.ndim)

# View the dimension of tensor
print(X_train.shape)
print(X_valid.shape)


# In[13]:


#each pixel has its unique color code . To perform Machine Learning, it is important to convert all 
#the values from 0 to 255 for every pixel to a range of values from 0 to 1. 
#The simplest way is to divide the value of every pixel by 255 to get the values in the range of 0 to 1.
#X_train = X_train / 255
#X_test = X_test / 255


# In[15]:


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


# In[16]:


#displays all the model’s layers
model.summary()


# In[17]:


model.layers


# In[18]:


#compiling the model 
#model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

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


# In[19]:


#model.fit(X_train, y_train, epochs=10, batch_size=100)
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_valid, y_valid), batch_size=100)


# In[20]:


#computes the loss and the accuracy of the model in the test set.
loss, acc = model.evaluate(x_test, y_test)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))


# In[21]:


train_acc = model.evaluate(X_train, y_train, verbose=1)
test_acc = model.evaluate(x_test, y_test, verbose=1)

print('Train set results:  ',train_acc)
print('Test set result.   ',test_acc)


# In[22]:


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


# In[23]:


#The neural network is trained. At each epoch during training, Keras displays
#the number of instances processed so far (along with a progress bar), the mean
#training time per sample, the loss and accuracy. we can see that the training loss
#went down, which is a good sign, and the validation accuracy reached 96% after 50
#epochs, not too far from the training accuracy, so there does not seem to be much
#overfitting going on

#Hence, after training the model we have achieved an accuracy of 99.88% for the training data set. 
#Now, it’s time to see how the model works in the test set and see whether we have achieved the required accuracy. 


# In[24]:


#plot the model 
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()


# In[25]:


#both the training and validation accuracy steadily increase during
#training, while the training and validation loss decrease
#validation curves are close to training curves---> not much overfitting 


# In[26]:


history.params


# In[27]:


#plotting 
print(history.history.keys())
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[28]:


#dictionary, containing the loss and extra metrics it
#measured at the end of each epoch on the training set and on the validation set

history.history


# In[29]:


test_loss, test_acc = model.evaluate(x_test, y_test)


# In[30]:


#Predictions 
# we can use the model’s method predict() to make predictions on new instances 
#Since we don’t have actual new instances, we will just use the first 10 instances of
#the test set.
X_new = x_test[:10]
y_proba = model.predict(X_new)
y_proba.round(3)
#y_proba


# #tuning

# In[31]:


#The flexibility of neural networks is also one of their main drawbacks: there are many
#hyperparameters to work with. Not only can you use any imaginable network architecture,
#but even in a simple MLP you can change the number of layers, the number of
#neurons per layer, the type of activation function to use in each layer and etc. 
#GridSearchCV or RandomizedSearchCV


# In[32]:


pip install scikeras


# In[33]:


#to wrap our Keras models in objects that mimic regular Scikit-Learn regressors--> we need to 
#to create a function that will build and compile a Keras model, given a set of hyperparameters (4,9)

from scikeras.wrappers import KerasClassifier, KerasRegressor


# In[40]:


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


# In[41]:


#create a KerasRegressor based on this build_model() function 
import warnings
warnings.filterwarnings('ignore')


# In[42]:


#The KerasRegressor object is a thin wrapper around the Keras model built using build_model()
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

#


# In[43]:


#Since there are many hyperparameters, it is preferable to use a randomized search
#rather than grid search.(4)


# In[44]:


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


# In[45]:


#accessing the best parameters found, the best score, and the trained Keras model.
rnd_search_cv.best_params_


# In[46]:


rnd_search_cv.best_score_


# In[47]:


#
model = rnd_search_cv.best_estimator_.model


# In[49]:


model.summary


# In[ ]:





# In[ ]:





# In[ ]:


#model.save("my_keras_model.h5")


# In[ ]:


#model = keras.models.load_model("my_keras_model.h5")


# In[ ]:


#number of hiden layers: 
#for many problems you can start with just one or two hidden layers and
#it will work just fine (e.g., you can easily reach above 97% accuracy on the MNIST
#dataset using just one hidden layer with a few hundred neurons, and above 98% accu
#racy
#using two hidden layers with the same total amount of neurons, in roughly the
#same amount of training time).

#number of neurons 
#the number of neurons in the input and output layers is determined by the
#type of input and output your task requires. For example, the MNIST task requires 28
#x 28 = 784 input neurons and 10 output neurons

#Just like for the number of layers, you can try increasing the number of neurons gradually
#until the network starts overfitting. In general you will get more bang for the
#buck by increasing the number of layers than the number of neurons per layer.
#Unfortunately, as you can see, finding the perfect amount of neurons is still somewhat difficult. 

#simpler approach is to pick a model with more layers and neurons than you
#actually need, then use early stopping to prevent it from overfitting (and other regu
#larization techniques, such as dropout.

#The learning rate is arguably the most important hyperparameter. In general, the
#optimal learning rate is about half of the maximum learning rate.
#simple approach for tuning the learning rate is to start with a large value that
#makes the training algorithm diverge, then divide this value by 3 and try again,
#and repeat until the training algorithm stops diverging. At that point, you gener
#ally
#won’t be too far from the optimal learning rate

#Batch size. 
#The batch size can also have a significant impact on your model’s performance
#and the training time. In general the optimal batch size will be lower than 32 (in
#April 2018, Yann Lecun even tweeted "Friends don’t let friends use mini-batches
#larger than 32“). A small batch size ensures that each training iteration is very
#fast, and although a large batch size will give a more precise estimate of the gradients,
#in practice this does not matter much since the optimization landscape is
#quite complex and the direction of the true gradients do not point precisely in
#the direction of the optimum. However, having a batch size greater than 10 helps
#take advantage of hardware and software optimizations, in particular for matrix
#multiplications, so it will speed up training.

#activation function. 
#activation function earlier in this chapter: in general,
#the ReLU activation function will be a good default for all hidden layers


# In[ ]:


#batch normalisation (4.p335)
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.BatchNormalization(),
keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
keras.layers.BatchNormalization(),
keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
keras.layers.BatchNormalization(),
keras.layers.Dense(10, activation="softmax")
])


# In[ ]:


predictions = model.predict(x_test)# The predict() method return a vector with the predictions for the whole dataset elements.


# In[ ]:





# In[ ]:


#what can be done to improve the models' performance 
#here are several tricks you can try to improve the performance of the model like:

    #Changing the learning rate in the optimizer.
    #Decreasing/Increasing the batch size for training.
    #Changing the number of Neurons in the hidden layers.
    #Changing the number of hidden layers, while remembering that a layer’s output is the subsequent layer’s input.
    #You can also try using another optimizer instead of Adam like RMSProp, Adagrad, etc. (7,8)

