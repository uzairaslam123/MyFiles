import scipy.io as sio
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

def read_dataset():

    mat = sio.loadmat('data16.mat')
    X = mat['y_out']
    Y = mat['y_new']
    print('The shape of Y before One hot encoding is: ', Y.shape)
    print('The Contents of Y before One hot encoding is: ', Y)
    print('The shape of X before One hot encoding is: ', X.shape)
    print('The Contents of X before One hot encoding is: ', X)
    return (X,Y)

# Load the data
X, Y = read_dataset()
print("Y is: ", Y)
print("X is: ", X)

# Convert the dataset into train and test datasets
train_x, test_x = X[0:-16000], X[-16000:]
train_y, test_y = Y[0:-16000], Y[-16000:]

print("train_x is : ", train_x)
print("test_x is: ", test_x)
print("train_y is : ", train_y)
print("test_y is: ", test_y)

#Create Model
model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=20, batch_size=80, verbose=2)


#To Print the Predicted and Expected (True) States
print("Entering the for loop: ")
predictions = list()
true_val = list()
true_val = test_y
item = 1
for i in range(len(test_x)):
    x = test_x[i, :]
    print(item,": The current output value is: ", x)
    yhat = model.predict_classes(x, batch_size=1)
    predictions.append(yhat)
    expected = test_y[i, :]
    item = item + 1
    print("Predicted State=%f, True State=%f" % (yhat,expected))

count = 0

for i in range (len(predictions)):

     if predictions[i] == true_val[i]:
         count = count + 1
     #print("Size of Count is: ", count)

total = count / len(predictions)
print ("Total Accuracy is: ", total*100)
