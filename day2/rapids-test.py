#!/usr/bin/env python
# coding: utf-8

import cudf
import numpy as np
import pandas as pd
from time import time

import cuml.neighbors
from cuml import __version__ as cuml_version

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import __version__ as sklearn_version
import sklearn.neighbors

print('Using cudf version:', cudf.__version__)
print('Using cuml version:', cuml_version)
print('Using sklearn version:', sklearn_version)
