#!/bin/bash -f
#SBATCH --job-name=xxz_1.5
#SBATCH --time=168:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rw2515@nyu.edu
#SBATCH --error=slurm.error
#SBATCH --output=slurm.out
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=180GB

module purge

# Set everything up
ulimit -u 10000
source ~/neural-env/bin/activate
python3 main_script.py
