#!/bin/bash
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --time=15:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --account=project_2003528
#xSBATCH --reservation=mlintro

module list

set -xv
srun singularity_wrapper exec python3 $*
