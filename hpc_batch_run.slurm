#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=sir_abm_mobility
#SBATCH --output sir_abm_mobility_%j.out
#SBATCH --error sir_abm_mobility_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=6GB
#SBATCH --export=ALL
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aogueda@gmu.edu

set echo
umask 0027

module load gnu12
source ~/ds-geo/bin/activate

python batch_run.py
