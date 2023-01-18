#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -c 50
#SBATCH --mem 100g
#SBATCH -t 24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=zhutchen@live.unc.edu

module add python/3.9.6
python3 get_table_optimization.py 
