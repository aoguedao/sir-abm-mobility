#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=sir_abm_mobility
#SBATCH --output sir_abm_mobility_%A_%a.out
#SBATCH --error sir_abm_mobility_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=4GB
#SBATCH --export=ALL
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aogueda@gmu.edu

set echo
umask 0027

module load gnu10
module load python
source ~/ds-geo/bin/activate

python batch_run.py
