from google.colab import drive
drive.mount('/content/drive')

!cp drive/MyDrive/resnets_utils.py resnets_utils.py

from resnets_utils import *
import tensorflow as tf
import numpy as np

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

!tar -xvf  /content/drive/MyDrive/sentences.tgz -C "/content/images"

from itertools import islice

form_writer = {}
forms_file_info = "/content/drive/MyDrive/forms_BA.txt"
with open(forms_file_info) as f:
  for line in islice(f, 0, None):
    line_list = line.split(' ')
    form_id = line_list[0]
    writer = line_list[1]
    form_writer[form_id] = writer

list(form_writer.items())[0:5]

# Select all writer
from collections import Counter

top_writers = []
num_writers = None
writers_counter = Counter(form_writer.values()) #counts number of every writer`s handwritting in order
for writer_id,_ in writers_counter.most_common(num_writers):
  top_writers.append(writer_id)

print(top_writers[0:5])

#Mapping the wirter id with their form id
top_forms = []
for form_id, author_id in form_writer.items():
  if author_id in top_writers:
    top_forms.append(form_id)

print(top_forms[0:5])

import glob
import shutil
import os

# Create temp directory to save writers' forms in (assumes files have already been copied if the directory exists)
temp_sentences_path = "/content/temp_sentences"
if not os.path.exists(temp_sentences_path):
  os.makedirs(temp_sentences_path)
  # Copy forms that belong to the most common writers to the temp directory
  original_sentences_path = "/content/images/**/**/*.png"
  for file_path in glob.glob(original_sentences_path):
    image_name = file_path.split('/')[-1]
    file_name, _ = os.path.splitext(image_name)
    form_id = '-'.join(file_name.split('-')[0:2])
    if form_id in top_forms:
      shutil.copy2(file_path, temp_sentences_path + "/" + image_name)

print(file_name)
print(form_id)

from keras.preprocessing import image

img_files = np.zeros([16752,64,64,1], dtype=np.float64)

img_targets = np.zeros((0), dtype=np.str) #Include the writer_id
img_files_path = np.zeros((0), dtype=np.str)
path_to_files = os.path.join(temp_sentences_path, '*')

for i, file_path in enumerate(glob.glob(path_to_files)):

  #print(len(glob.glob(path_to_files)))

  img = image.load_img(file_path, color_mode="grayscale", target_size=(64, 64))

  #Saving the location of images.
  img_files_path = np.append(img_files_path, file_path)

  #Making the images ready include normalization.
  img_arr = image.img_to_array(img)
  #img_arr = np.expand_dims(img_arr, axis=0) # a new dimension added to img_arr
  img_arr = img_arr/255.0

  img_files[i] = img_arr

  file_name, _ = os.path.splitext( file_path.split('/')[-1] )
  form_id = '-'.join(file_name.split('-')[0:2])
  for key in form_writer:
    if key == form_id:
      img_targets = np.append(img_targets, form_writer[form_id])

print(img_files_path[0])
print(img_targets[0:5])

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

file_name = img_files_path[0]
img = mpimg.imread(file_name)
plt.figure(figsize = (10,10))
plt.imshow(img, cmap ='gray')

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(img_targets)
encoded_img_targets = encoder.transform(img_targets)

print("Writer ID        : ", img_targets[:2])
print("Encoded writer ID: ", encoded_img_targets[:2])

""" ## CNN"""

from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from keras.layers import Input, Add,Dropout, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D

def identity_block(X, f, filters, training=True, initializer=random_uniform):
  # Retrieve Filters
  F1, F2, F3 = filters

  # Save the input value. We'll need this later to add back to the main path.
  X_shortcut = X
  cache = []
  # First component of main path
  X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
  X = BatchNormalization(axis = 3)(X, training = training) # Default axis
  X = Activation('relu')(X)
  X = Dropout(0.2)(X)

  # Second component of main path
  X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
  X = BatchNormalization(axis = 3)(X, training = training)
  X = Activation('relu')(X)
  X = Dropout(0.2)(X)

  # Third component of main path
  X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
  X = BatchNormalization(axis = 3)(X, training = training)

  # Final step: Add shortcut value to main path, and pass it through a RELU activation
  X = Add()([X, X_shortcut])
  X = Activation('relu')(X)

  return X

def convolutional_block(X, f, filters, s = 2, training=True, initializer=glorot_uniform):

  # Retrieve Filters
  F1, F2, F3 = filters
  # Save the input value
  X_shortcut = X
  # First component of main path glorot_uniform(seed=0)
  X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid',
             kernel_initializer = initializer(seed=0))(X)
  X = BatchNormalization(axis = 3)(X, training=training)
  X = Activation('relu')(X)

  # Second component of main path
  X = Conv2D(filters =F2,  kernel_size = f, strides = (1,1), padding = 'same',
             kernel_initializer = initializer(seed=0))(X)
  X = BatchNormalization(axis = 3)(X , training=training)
  X = Activation('relu')(X)
  X = Dropout(0.2)(X)

  # Third component of main path
  X = Conv2D(filters =F3, kernel_size = 1, strides = (1,1), padding = 'valid',
             kernel_initializer = initializer(seed=0))(X)
  X = BatchNormalization(axis = 3)(X , training=training)
  X = Dropout(0.2)(X)

  # SHORTCUT PATH
  X_shortcut = Conv2D(filters =F3, kernel_size = 1, strides = (s,s), padding = 'valid',
                      kernel_initializer = initializer(seed=0))(X_shortcut)
  X_shortcut = BatchNormalization(axis = 3)(X_shortcut , training=training)

  # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
  X = Add()([X, X_shortcut])
  X = Activation('relu')(X)

  return X

from keras import layers
from tensorflow.keras import regularizers
from keras.models import Model

def ResNet18(input_shape , classes ):


  # Define the input as a tensor with shape input_shape
  X_input = Input(input_shape)

  # Zero-Padding
  X = ZeroPadding2D((3, 3))(X_input)

  # Stage 1
  X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis = 3)(X)
  X = Activation('relu')(X)
  X = layers.ZeroPadding2D((1, 1))(X)
  X = MaxPooling2D((3, 3), strides=(2, 2))(X)
  X = Dropout(0.2)(X)

  # Stage 2
  X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
  X = identity_block(X, 3, [64, 64, 256])

  # Stage 3
  X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
  X = identity_block(X, 3, [128, 128, 512])

  # Stage 4
  X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
  X = identity_block(X, 3, [256, 256, 1024])

  # Stage 5
  X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
  X = identity_block(X, 3, [512, 512, 2048])


  # AVGPOOL
  X = AveragePooling2D((2, 2), name='avg_pool')(X)

  # Include dropout with probability of 0.2 to avoid overfitting
  X = Dropout(0.2)(X)

  # Output layer
  regularizers.l2(l2=0.02)
  X = Flatten()(X)
  X = Dense(classes, activation='softmax' , kernel_regularizer='l2', kernel_initializer = glorot_uniform(seed=0))(X)

  # Create model
  model = Model(inputs = X_input, outputs = X)

  return model

model = ResNet18(input_shape = (64, 64, 1), classes = 657)
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.resnet50 import preprocess_input


train_datagen = ImageDataGenerator( zoom_range = 0.2 , horizontal_flip=True, fill_mode="nearest", vertical_flip = True,rotation_range=5, shear_range=0.2)

kf = KFold(n_splits = 10, shuffle=True)
kf.get_n_splits(img_files)
acc_per_fold = []
loss_per_fold = []
history_list = list()
for k, (train_index, test_index) in enumerate(kf.split(img_files)):

  X_train, X_test = img_files[train_index], img_files[test_index]
  y_train, y_test = encoded_img_targets[train_index], encoded_img_targets[test_index]

  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle = True)

  y_train = convert_to_one_hot(y_train, 657).T
  y_test = convert_to_one_hot(y_test, 657).T
  y_val = convert_to_one_hot(y_val, 657).T

  train_datagen.fit(X_train)
  train_datagen.fit(X_val)
  print('------------------------------------------------------------------------')
  print(f'Training for fold {k} ...')
  history = model.fit( train_datagen.flow(X_train, y_train, batch_size=128), epochs = 80 , validation_data=(train_datagen.flow(X_val,y_val)) )
  history_list.append(history)

  scores = model.evaluate(X_test, y_test, verbose=0)
  print(f'Score for fold {k}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])



# summarize history for accuracy
fig, axs = plt.subplots(len(history_list), sharex=True,figsize=(10,10))

fig.suptitle("\n".join(["Models Accuracy"]), y=1.01, weight='bold')
line_labels = ["TrainAccuracy","ValidationAccuracy"]
line_objects = []


for i in range(len(history_list)):
  axs[i].set_title("\n".join([f"Fold{i}"]))
  line_objects = axs[i].plot(history_list[i].history['accuracy'])
  axs[i].plot(history_list[i].history['val_accuracy'])
  fig.tight_layout()


fig.legend(line_objects, labels=line_labels, loc="upper right", borderaxespad=0.1)

fig.text(0.5, 0.001, 'Epochs', ha='center', weight='bold')
fig.text(0.002, 0.5, 'Accuracy', va='center', rotation='vertical', weight='bold')

# summarize history for loss
fig, axs = plt.subplots(len(history_list), sharex=True,figsize=(10,10))

fig.suptitle("\n".join(["Models Loss"]), y=1.01, weight='bold')
line_labels = ["TrainLoss","ValidationLoss"]
line_objects = []


for i in range(len(history_list)):
  axs[i].set_title("\n".join([f"Fold{i}"]))
  line_objects = axs[i].plot(history_list[i].history['loss'])
  axs[i].plot(history_list[i].history['val_loss'])
  fig.tight_layout()

fig.legend(line_objects, labels= line_labels, loc="upper right", borderaxespad=0.1)

fig.text(0.5, 0.001, 'Epochs', ha='center', weight='bold')
fig.text(0.002, 0.5, 'Loss', va='center', rotation='vertical', weight='bold')


plt.show()

weight_accuracy = np.array([ 0.9, 1.1, 1.4, 1.7, 2])
weight_loss = np.array([0.9, 1.1, 1.4, 1.7, 2])

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Test Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average test scores for all folds:')
print(f'> Accuracy: {np.average(acc_per_fold, weights= weight_accuracy)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.average(loss_per_fold, weights= weight_loss)}')
print('------------------------------------------------------------------------')

acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
