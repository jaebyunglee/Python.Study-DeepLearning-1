# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 19:15:05 2019

@author: User
"""


from tensorflow.keras import layers, models
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 #opencv 설치

tf.random.set_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#train_dir = os.path.join('C:/Users/User/Desktop/jb/python/images/TRAIN')
#val_dir = os.path.join('C:/Users/User/Desktop/jb/python/images/Valid')
#test_dir = os.path.join('C:/Users/User/Desktop/jb/python/images/TEST')


##############################################################################
###                          train data                                    ###
##############################################################################
#TRAIN_DATADIR = "C:/Users/User/Desktop/jb/python/images/TRAIN"
TRAIN_DATADIR = '/mnt/iamsheep/jaebyung/images/TRAIN'
CATEGORIES = ['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']

IMG_SIZE = 224
CHANNEL = 3
#### training data plt
#for category in CATEGORIES:
#    path = os.path.join(TRAIN_DATADIR, category)
#    for img in os.listdir(path):
#        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
#        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#        plt.imshow(new_array)
#        plt.show()
#        break
#    break


################ create training data#########################################
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(TRAIN_DATADIR, category) #path
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()
len(training_data)
import random
random.shuffle(training_data)
###############################################################################
TRAIN_X = []
TRAIN_y = []

for features, label in training_data:
    TRAIN_X.append(features)
    TRAIN_y.append(label)    

TRAIN_X = np.array(TRAIN_X).reshape(-1, IMG_SIZE, IMG_SIZE ,CHANNEL)
TRAIN_y = np.array(TRAIN_y)
##############################################################################


##############################################################################
###                          valid data                                    ###
##############################################################################
#VALID_DATADIR = "C:/Users/User/Desktop/jb/python/images/Valid"
VALID_DATADIR = '/mnt/iamsheep/jaebyung/images/Valid'
CATEGORIES = ['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']


#### training data plt
#for category in CATEGORIES:
#    path = os.path.join(VALID_DATADIR, category)
#    for img in os.listdir(path):
#        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
#        plt.imshow(img_array)
#        plt.show()
#        break
#    break


################ create training data#########################################
valid_data = []


def create_valid_data():
    for category in CATEGORIES:
        path = os.path.join(VALID_DATADIR, category) #path
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                valid_data.append([new_array, class_num])
            except Exception as e:
                pass
create_valid_data()
len(valid_data)
import random
random.shuffle(valid_data)
###############################################################################
VALID_X = []
VALID_y = []

for features, label in valid_data:
    VALID_X.append(features)
    VALID_y.append(label)    

VALID_X = np.array(VALID_X).reshape(-1, IMG_SIZE, IMG_SIZE ,CHANNEL)
VALID_y = np.array(VALID_y)
##############################################################################


##############################################################################
###                           test data                                    ###
##############################################################################
#TEST_DATADIR = "C:/Users/User/Desktop/jb/python/images/TEST"
TEST_DATADIR = '/mnt/iamsheep/jaebyung/images/TEST'
CATEGORIES = ['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']


#### training data plt
#for category in CATEGORIES:
#    path = os.path.join(TEST_DATADIR, category)
#    for img in os.listdir(path):
#        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
#        plt.imshow(img_array)
#        plt.show()
#        break
#    break


################ create training data#########################################
test_data = []


def create_test_data():
    for category in CATEGORIES:
        path = os.path.join(TEST_DATADIR, category) #path
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])
            except Exception as e:
                pass
create_test_data()
len(test_data)
import random
random.shuffle(test_data)
###############################################################################
TEST_X = []
TEST_y = []

for features, label in test_data:
    TEST_X.append(features)
    TEST_y.append(label)    

TEST_X = np.array(TEST_X).reshape(-1, IMG_SIZE, IMG_SIZE ,CHANNEL)
TEST_y = np.array(TEST_y)

x_train = tf.convert_to_tensor(TRAIN_X, dtype=tf.float32) / 255.
x_valid = tf.convert_to_tensor(VALID_X, dtype=tf.float32) / 255.
x_test = tf.convert_to_tensor(TEST_X, dtype=tf.float32) / 255.


##############################################################################
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import  Dropout, Input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

vgg16_model = tensorflow.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
#let’s disable training on all but the last 4 layers of the pretrained model.
for layer in vgg16_model.layers[:-4]:
    layer.trainable = False
    
    
    
    # Create the model
model = Sequential()
 
# Add the vgg convolutional base model
model.add(vgg16_model)
 
# Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation="softmax"))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()
# Convert from uint8 to float32 and
# normalize images value from [0, 255] to [0, 1].



# Cross-Entropy loss function.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


# Parameters for Training
learning_rate = 0.0001
batch_size = 16
EPOCHS = 10

# Stochastic gradient descent optimizer.
optimizer = Adam(learning_rate)


# Use tf.data API to shuffle and batch data.
train_ds = tf.data.Dataset.from_tensor_slices((x_train, TRAIN_y)).shuffle(10000).batch(batch_size)
valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, VALID_y)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, TEST_y)).batch(batch_size)


#train_data = tf.data.Dataset.from_tensor_slices((x_train, TRAIN_y))
#training_batch = train_data.batch(batch_size).repeat(training_steps)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#train step function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

#test step function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
  
  
for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
  