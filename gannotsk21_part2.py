import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import tensorflow as tf

big_list = [] 
#read data from file
f = open('histo.txt')
for line in f:
    big_list.append(line.strip().split(" "))
f.close()

#separating histogram into testing and training sets
yes_list = big_list[0:1505]
no_list = big_list[1505:5259]
unsure_list = big_list[5259:]

yes_train = yes_list[0:int(1505 * 0.8)]
yes_test= yes_list[int(1505 * 0.8): 1505]
no_train = no_list[0: int((5259 - 1505) * .8)]
no_test = no_list[int((5259 - 1505) * .8): 5259]

train_data = np.concatenate((yes_train, no_train))
test_data = np.concatenate((yes_test, no_test))
np.random.shuffle(train_data)
np.random.shuffle(test_data)



#separating values into x and y values
train_X=train_data[:,1:769]
train_Y=train_data[:,0]
train_Y=[[x] for x in train_Y]
print("Training data")
print(train_X[0:5])
print(train_Y[0:5])
test_X=test_data[:,1:769]
test_Y=test_data[:,0]
test_Y=[[x] for x in test_Y]

num_classes=1
epochs=20
batch_size=100
