#!/bin/bash
#SBATCH --ntasks=1                          # Request number of CPU cores
#SBATCH --time=0-00:10:00                   # Job should run for time
#SBATCH --account=wetosa                 # Accounting
#SBATCH --job-name=debug1   	            # Job name
#SBATCH --output=debug.out 	                # %j will be replaced with job ID
#SBATCH --partition=debug

source $HOME/.bash_profile
source $HOME/.bashrc
cd /home/pjstanle/spatial-optimization

conda activate spatial

python optimize_farm.py 1 0 $SLURM_ARRAY_TASK_ID aep 1