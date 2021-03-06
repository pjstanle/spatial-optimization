#!/bin/bash
#SBATCH --ntasks=1                          # Request number of CPU cores
#SBATCH --time=02-00:00:00                   # Job should run for time
#SBATCH --account=wetosa                    # Accounting
#SBATCH --job-name=aep	                # Job name
#SBATCH --mail-user pj.stanley@nrel.gov     # user email for notifcations
#SBATCH --mail-type FAIL                    # ALL will notify for BEIGN,END,FAIL

source $HOME/.bash_profile
source $HOME/.bashrc
cd /home/pjstanle/spatial-optimization

conda activate spatial

python optimize_mesh.py 1 1.1 3 aep
# python optimize_remove.py $1 $2 $3 $4
