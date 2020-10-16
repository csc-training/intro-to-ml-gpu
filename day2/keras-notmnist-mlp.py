#!/usr/bin/env python
# coding: utf-8

# # notMNIST handwritten digits classification with MLPs
#
# In this script, we'll train a multi-layer perceptron model to
# classify notMNIST digits using TensorFlow with Keras (tf.keras)


import sys
import numpy as np
#import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from distutils.version import LooseVersion as LV

from pml_utils import get_notmnist, show_failures

print('Using Tensorflow version: {}, and Keras version: {}.'.format(
    tf.__version__, tf.keras.__version__))
assert(LV(tf.__version__) >= LV("2.0.0"))

# Check if we have GPU available

if tf.test.is_gpu_available():
    from tensorflow.python.client import device_lib
    for d in device_lib.list_local_devices():
        if d.device_type == 'GPU':
            print('GPU', d.physical_device_desc)
else:
    print('No GPU, using CPU instead.')


# ## notMNIST data set
#
# Next we'll load the notMNIST data set. First time we may have to
# download the data, which can take a while.


DATA_DIR = '/scratch/project_2003528/data/notMNIST/'
X_train, y_train, X_test, y_test = get_notmnist(DATA_DIR)

nb_classes = 10

X_train /= 255
X_test /= 255

# one-hot encoding:
Y_train = to_categorical(y_train.view(np.int32)-ord('A'), nb_classes)
Y_test = to_categorical(y_test.view(np.int32)-ord('A'), nb_classes)

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape, X_train.dtype)
print('y_train:', y_train.shape, y_train.dtype)
print('Y_train:', Y_train.shape, Y_train.dtype)


# ## Multi-layer perceptron (MLP) network
#
# Let's now create an MLP model that has multiple layers, non-linear
# activation functions, and dropout layers. `Dropout()` randomly sets
# a fraction of inputs to zero during training, which is one approach
# to regularization and can sometimes help to prevent overfitting.
#
# There are two options below, a simple and a bit more complex model.
# Select either one.
#
# The output of the last layer needs to be a softmaxed 10-dimensional
# vector to match the groundtruth (`Y_train`).
#
# Finally, we again `compile()` the model, using Adam
# (https://keras.io/optimizers/#adam) as the optimizer.

# Model initialization:
model = Sequential()

# A simple model:
model.add(Dense(units=20, input_dim=28*28))
model.add(Activation('relu'))

# A bit more complex model:
# model.add(Dense(units=50, input_dim=28*28))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))

# model.add(Dense(units=50))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))

# The last layer needs to be like this:
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())


# ### Training
epochs = 10

history = model.fit(X_train.reshape((-1,28*28)),
                    Y_train,
                    epochs=epochs,
                    batch_size=128,
                    verbose=2)

# ### Inference
#
# Accuracy for test data.

scores = model.evaluate(X_test.reshape((-1, 28*28)), Y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X_test.reshape((-1, 28*28)))

predictions = np.argmax(predictions, axis=1)
predictions = [chr(x) for x in predictions+ord('A')]
predictions = np.array(predictions)

print('Predicted', len(predictions), 'letters with accuracy:',
      accuracy_score(y_test, predictions))
print()

# #### Confusion matrix, accuracy, precision, and recall
#
# We can also compute the confusion matrix to see which digits get
# mixed the most, and look at classification accuracies separately for
# each class.

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
print('Confusion matrix (rows: true classes; columns: predicted classes):')
print()
cm = confusion_matrix(y_test, predictions, labels=labels)
# df_cm = pd.DataFrame(cm, columns=labels, index=labels)
# print(df_cm.to_string())
print()

print('Classification accuracy for each class:')
print()
for i, j in enumerate(cm.diagonal()/cm.sum(axis=1)):
    print("%s: %.4f" % (labels[i], j))

# Precision and recall for each class:

print()
print('Precision and recall for each class:')
print()
print(classification_report(y_test, predictions, labels=labels))
