#!/bin/bash -f
#SBATCH --job-name=tfi_3.0
#SBATCH --time=168:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rw2515@nyu.edu
#SBATCH --error=slurm.error
#SBATCH --output=slurm.out
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=180GB

# Set everything up
module purge
ulimit -u 10000
source ~/neural-env/bin/activate
python3 tester.py
