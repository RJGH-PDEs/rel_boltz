#!/bin/bash

#SBATCH -N 1
#SBATCH -n 128
#SBATCH -o n5_tensor.%j.out
#SBATCH -e n5_tensor.%j.err
#SBATCH -J boltz_n5
#SBATCH -p normal
#SBATCH -A DMS23021
#SBATCH -t 2:00:00

#SBATCH --mail-user=rodrigogonzalez@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# rel_boltz scripts use relative paths (see CLAUDE.md "Running things"),
# so this must run with cwd = src/, not the repo root.
cd /work/09611/rodrigojosegonzalez/ls6/research/rel_boltz/src

source ~/.bashrc
conda activate ttenv
python3 run_n5_tensor.py
