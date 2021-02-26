#!/bin/bash
#SBATCH --ntasks=1                          # Request number of CPU cores
#SBATCH --time=5-00:00:00                   # Job should run for time
#SBATCH --account=wetosa                    # Accounting
#SBATCH --job-name=spatial	                # Job name
#SBATCH --mail-user pj.stanley@nrel.gov     # user email for notifcations
#SBATCH --mail-type FAIL                    # ALL will notify for BEIGN,END,FAIL

source $HOME/.bash_profile
source $HOME/.bashrc
cd /home/pjstanle/spatial-optimization

conda activate spatial

python optimize_farm.py 3 0 $SLURM_ARRAY_TASK_ID coe 0
