#!/bin/bash
##################################################
# Sets up environment and starts standard interactive session
##################################################

# Load modules
module load gcc/7.1.0-fasrc01 cdo/1.9.4-fasrc02
module load python/3.6.3-fasrc02
# load project environment
source activate climate_odyssey
