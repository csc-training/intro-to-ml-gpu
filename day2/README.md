# Day 2

## Exercise sessions

### Exercise 5

### Exercise 6

Classification with random forest and gradient boosted trees.

* *rapids-notmnist-rf.py*: notMNIST classification with random forest
* *rapids-notmnist-xgb.py*:notMNIST classification with XGBoost

### Exercise 7

Classification with neural networks.

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

   or for PyTorch:
   
        module purge
        module load pytorch/1.3.0

## Edit and submit jobs

1. Edit and submit jobs:

        nano rapids-test.py  # or substitute with your favorite text editor
        sbatch run.sh rapids-test.py  # when using a training account

   There is a separate slurm script for PyTorch, e.g.:
   
        sbatch run-pytorch.sh pytorch_dvc_cnn_simple.py

   You can also specify additional command line arguments, e.g.

        sbatch run.sh tf2-dvc-cnn-evaluate.py dvc-cnn-simple.h5

2. See the status of your jobs or the queue you are using:

        squeue -l -u trainingxxx
        squeue -l -p gpu

3. After the job has finished, examine the results:

        less slurm-xxxxxxxx.out

