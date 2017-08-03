#!/bin/bash
#SBATCH --job-name=joint-correlation-scaling
#SBATCH --output=jointcs.out
#SBATCH --ntasks=32
#SBATCH --time=48:00:00

nohup mpirun python ./joint-correlation-scaling.py
