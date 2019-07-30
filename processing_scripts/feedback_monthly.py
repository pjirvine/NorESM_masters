
"""
Script to mergetime on all feedback monthly data
"""

# Import modules

from cdo import *
cdo = Cdo()
import my_shell_tools
import os
import os.path
import CMIP5_functions as cmip5

# import functions from file
from GLENS_functions import *

"""
Define variables which specify which files to process
"""

raw_monthly_dir = "/n/home03/pjirvine/keithfs1_pji/GLENS/raw_monthly_data/"
temp_dir = "/n/home03/pjirvine/keithfs1_pji/GLENS/temp_merged/"

runs = [str(IDX+1).zfill(3) for IDX in xrange(21)]
variables = ['LHFLX','PRECC','PRECL','PRECTMX','TREFHT','TREFHTMX']

seach_format = "feedback.{run}.cam.h0.{var}."
dates = '202001-209912'

"""
Main loop
"""

for var in variables:
    for run in runs:
        
        run_name = seach_format.format(run=run, var=var)
        years_out = merge_years(run_name, dates)
        
        print run_name, years_out
        
#end