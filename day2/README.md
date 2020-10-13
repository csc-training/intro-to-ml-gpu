# Day 2

## Exercise sessions

### Exercise 5

Classification with nearest neighbors, comparing Scikit-learn (CPU) with RAPIDS (GPU) implementations.

* *rapids-notmnist-nn.py*: notMNIST classification with nearest neighbours

### Exercise 6

Classification with random forest and gradient boosted trees.

* *rapids-notmnist-rf.py*: notMNIST classification with random forest
* *rapids-notmnist-xgb.py*:notMNIST classification with XGBoost

### Exercise 7

Classification with neural networks using Keras (TensorFlow 2).

* *keras-notmnist-mlp.py*: notMNIST classification using MLP neural network

### Exercise 8

Clustering and data visualization.

This exercise requires a GPU-accelerated Jupyter Notebooks instance.

* Exercise 08a: [Clustering](Exercise-08a.ipynb)
* Exercise 08b: [Data visualization](Exercise-08b.ipynb)

## Setup

1. Login to Puhti using a training account (or your own CSC account):

        ssh -l trainingxxx puhti.csc.fi

2. Clone and cd to the exercise repository:

        git clone https://github.com/csc-training/intro-to-ml-gpu.git
        cd intro-to-ml-gpu/day2

3. Set up the module environment for Rapids

        module purge
        module load rapids/0.15-sng

   or for Keras:
   
        module purge
        module load tensorflow/2.0.0

## Edit and submit jobs

1. Edit and submit jobs:

        nano rapids-test.py  # or substitute with your favorite text editor
        sbatch run.sh rapids-test.py  # when using a training account

   There is a separate slurm script for Keras (used in Exercise 7), e.g.:
   
        sbatch run-keras.sh keras-notmnist-mlp.py

2. See the status of your jobs or the queue you are using:

        squeue -l -u $USER
        squeue -l -p gpu

3. After the job has finished, examine the results:

        less slurm-xxxxxxxx.out

