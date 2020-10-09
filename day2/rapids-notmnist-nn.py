#!/usr/bin/env python
# coding: utf-8

# notMNIST letters classification with nearest neighbors

# In this script, we'll use nearest-neighbor classifiers
# (https://docs.rapids.ai/api/cuml/stable/api.html#id18) to classify
# notMNIST letters using a GPU and RAPIDS (https://rapids.ai/)
# libraries (cudf, cuml).
#
# **Note that a GPU is required with this notebook.**
#
# This version of the notebook has been tested with RAPIDS version
# 0.15.

# First, the needed imports.


import cudf
import numpy as np
import pandas as pd
from time import time

import cuml.neighbors
from cuml import __version__ as cuml_version

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import __version__ as sklearn_version
import sklearn.neighbors

import sys
sys.path.append('../')

from pml_utils import get_notmnist

print('Using cudf version:', cudf.__version__)
print('Using cuml version:', cuml_version)
print('Using sklearn version:', sklearn_version)

# Then we load the notMNIST data. First time we need to download the
# data, which can take a while. The data is stored as Numpy arrays in
# host (CPU) memory.

print()
print('Loading data begins')
t0 = time()

DATA_DIR = '/scratch/project_2003528/data/notMNIST/'
X_train, y_train, X_test, y_test = get_notmnist(DATA_DIR)

print()
print('notMNIST data loaded: train:', len(X_train), 'test:', len(X_test))
print('X_train:', type(X_train), 'shape:', X_train.shape)
print('y_train:', type(y_train), 'shape:', y_train.shape)
print('X_test:', type(X_test), 'shape:', X_test.shape)
print('y_test:', type(y_test), 'shape:', y_test.shape)

print('Loading data done in {:.2f} seconds'.format(time()-t0))


# Let's create first a 1-NN classifier with scikit-learn, using CPU
# only. Note that with nearest-neighbor classifiers there is no
# internal (parameterized) model and therefore no learning required.
# Instead, calling the `fit()` function simply stores the samples of
# the training data in a suitable data structure. Unfortunately, the
# dataset is so large that simply creating the data structure is still
# quite slow on the CPU. Therefore, we limit the training set to
# 50,000 items so we won't have to wait for too long...

print()
print('Creating 1-NN classifier on CPU with 50k training subset begins')
t0 = time()

n_neighbors = 1
clf_nn = sklearn.neighbors.KNeighborsClassifier(n_neighbors)
clf_nn.fit(X_train[:50000], y_train[:50000])

print('Creating 1-NN classifier on CPU took {:.2f} seconds'.format(time()-t0))

# Next, we'll classify 200 samples with it

print()
print('Starting CPU inference')
t0 = time()

pred_nn = clf_nn.predict(X_test[:200, :])

print('Inference on CPU with 200 samples took {:.2f} seconds'.format(time()-t0))

print('Predicted', len(pred_nn), 'digits with accuracy:',
      accuracy_score(y_test[:len(pred_nn)], pred_nn))

# Let's convert our training data to cuDF DataFrames in device (GPU)
# memory. We will also convert the classes in `y_train` to integers in
# [0..9].
#
# We do not explicitly need to convert the test data as the GPU-based
# inference functionality will take care of it.

print()
print('Copying data to GPU begins')
t0 = time()

cu_X_train = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
cu_y_train = cudf.Series(y_train.view(np.int32)-ord('A'))

print('cu_X_train:', type(cu_X_train), 'shape:', cu_X_train.shape)
print('cu_y_train:', type(cu_y_train), 'shape:', cu_y_train.shape)

print('Copying data to GPU done in {:.2f} seconds'.format(time()-t0))

# Now we will create 1-NN classifier on GPU using RAPIDS with the full 500k dataset

print()
print('Creating 1-NN classifier on GPU with full 500k training subset begins')
t0 = time()

n_neighbors = 1
cu_clf = cuml.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
cu_clf.fit(cu_X_train, cu_y_train)

print('Creating 1-NN classifier on GPU took {:.2f} seconds'.format(time()-t0))

# ### Inference
#
# We will use GPU-based inference to predict the classes for the test
# data.

print()
print('Inference begins')
t0 = time()

predictions = cu_clf.predict(X_test, predict_model='GPU').values_host.flatten()
predictions = [chr(x) for x in predictions+ord('A')]
predictions = np.array(predictions)

print('Inference done in {:.2f} seconds'.format(time()-t0))
print()

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
df_cm = pd.DataFrame(cm, columns=labels, index=labels)
print(df_cm.to_string())
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
