#!/bin/bash
#
#SBATCH -J ann_mean_std
#SBATCH -p shared                # partition (queue)
#SBATCH -N 1                      # number of nodes
#SBATCH -n 1                      # number of cores
#SBATCH --mem=3000                 # memory pool for all cores
#SBATCH -t 0-00:30                 # time (D-HH:MM)
#SBATCH -o /n/home03/pjirvine/slurm/out/SLURM.%j.%N.out        # STDOUT
#SBATCH -e /n/home03/pjirvine/slurm/err/SLURM.%j.%N.err        # STDERR
#SBATCH --mail-type=ALL
#                               Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=p.j.irvine@gmail.com
#                               Email to which notifications will be sent


python ann_mean_std.py

############################## slurm ends ##############################
