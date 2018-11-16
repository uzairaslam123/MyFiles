from random import randint
from numpy import array
import scipy.io as sio
import numpy as np
from pandas import Series
from numpy import argmax
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
from matplotlib import pyplot
import random
import datetime
from random import randint
from numpy import array
from numpy import argmax
from pandas import concat
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout


def read_dataset():

    mat = sio.loadmat('data17.mat')
    X = mat['Input']
    Y = mat['x_before']
    print('The shape of Y before One hot encoding is: ', Y.shape)
    print('The Contents of Y before One hot encoding is: ', Y)
    print('The shape of X before One hot encoding is: ', X.shape)
    print('The Contents of X before One hot encoding is: ', X)

    #encoder = LabelEncoder()
    #encoder.fit(y)
    #y = encoder.transform(y)
 # Encode the dependent variable
    #Y = one_hot_encode(y)
    #print("The shape of Y after One Hot Encoding is : ", Y.shape)
    #print('The Contents of Y after One hot encoding is: ', Y)
    return (X,Y)


# Define the encoder function.
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# Load the data
X, Y = read_dataset()
print("Y is: ", Y)
print("X is: ", X)


#X_Input = X[:, 0:2]
#Y_Output = Y[:, 2]

print("The Shape X is :", X.shape)
#print("The X_Input after splitting is :", X_Input)
print("The Shape of Y is :", Y.shape)
#print("The Y_Ouput after splitting is :", Y_Output)


# Convert the dataset into train and test datasets
train_x, test_x = X[0:-16000], X[-16000:]
train_y, test_y = Y[0:-16000], Y[-16000:]
test_x = test_x[:, 0:1]
print("The Type of test_x is :", type(test_x))

print("train_x is : ", train_x.shape)
print("test_x is: ", test_x.shape)
print("train_y is : ", train_y.shape)
print("test_y is: ", test_y.shape)
print("Creating the Model Now...")

#Create Model
model = Sequential()
model.add(Dense(20, input_dim=2, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(18, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='softmax'))
#print(model.summary())
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=20, batch_size=80, verbose=2)

#To check the accuracy of the model on test data.
#score = model.evaluate(test_x, test_y, verbose=2)
#predictions = model.predict_classes(test_x, batch_size=80)

#rounded = [round(x[0]) for x in predictions]
#print("Test Loss: ", score[0])
#print("Test Loss: ", score[1])

#To Print the Predicted and Expected (True) States
print("Entering the for loop: ")
predictions = np.zeros((len(test_x), 1))
p = np.array(len(test_x))
#N_yhat = np.array(len(test_x))
N_yhat = list()
print("The Shape of Predictions is :", predictions.shape)
true_val = np.array(len(test_y))
true_val = test_y
item = 1
for i in range(len(test_x)-1):
    print("The value of i is :", i)
    x = test_x[i, :]
    if i == 0:
        p = predictions[i, :]
    else:
        p = yhat

    print("The shape of p inside for loop is :", p.shape)
    #print("The Type of x inside for loop is :", type(x))
    print(item,": The current output value is: ", x)
    print("The shape of X is :", x.shape)
    x = np.concatenate((x, p))
    x = np.reshape(x, (1, 2))
    print("The shape of x after concatenation is :", x.shape)
    print("The value of x after concatenation is: ", x)
    yhat = model.predict_classes(x, batch_size=1)
    #print("The Type of yhat is :", type(yhat))
    #predictions = yhat
    print("The value of yhat after updating is: ", yhat)
    #p = np.concatenate((p, yhat))
    expected = test_y[i, :]
    item = item + 1
    N_yhat.append(yhat)
    print("Predicted State=%f, True State=%f" % (yhat,expected))


count = 0

print("Yhat is :", N_yhat)
N_yhat = np.reshape(N_yhat, (len(N_yhat), 1))
print("Shape of Yhat is :", N_yhat.shape)
print("After Reshaping N_Yhat is :", N_yhat)
print("Shape of true_val is :", true_val.shape)
print("True_val is :", true_val)


for i in range(len(N_yhat)):
    #print("Inside For Loop for Testing Accuracy")
    if N_yhat[i] == true_val[i]:
         count = count + 1
         #print("Inside If Statement of Testing Accuracy For Loop")
     #print("Size of Count is: ", count)

total = count / len((N_yhat))
print ("Total Accuracy is: ", total*100)



