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
Define variables
"""

run = '001'

feedback_name = "feedback.{run}.cam.h0.{var}.202001-209912{append}.nc"
control_name = "control.{run}.cam.h0.{var}.201001-209912{append}.nc"

in_dir = "/n/home03/pjirvine/keithfs1_pji/GLENS/combined_monthly_data/"
out_dir = "/n/home03/pjirvine/keithfs1_pji/GLENS/GEOL0013_seasons/"

mean_vars = ['TREFHT','PRECT','P-E']
max_vars = ['TREFHTMX','PRECTMX']

"""
Create a function to CDO process the seasonal and annual timeseries
"""

def cdo_all_seas(filename, var, run, mean=True):
    
    infile = in_dir + filename.format(var=var, run=run, append='')
    
    if mean:
        cdo.yearmean(input = infile, output = out_dir + filename.format(var=var, run=run, append='.ann'))
        seas_temp = cdo.seasmean(input = infile)
    else:
        cdo.yearmax(input = infile, output = out_dir + filename.format(var=var, run=run, append='.ann'))
        seas_temp = cdo.seasmax(input = infile)
        
    cdo.selseas('DJF', input = seas_temp, output = out_dir + filename.format(var=var, run=run, append='.djf'))
    cdo.selseas('MAM', input = seas_temp, output = out_dir + filename.format(var=var, run=run, append='.mam'))
    cdo.selseas('JJA', input = seas_temp, output = out_dir + filename.format(var=var, run=run, append='.jja'))
    cdo.selseas('SON', input = seas_temp, output = out_dir + filename.format(var=var, run=run, append='.son'))

# Loop over all mean_vars
for var in mean_vars:
    cdo_all_seas(control_name, var, run, mean=True)
    cdo_all_seas(feedback_name, var, run, mean=True)
    
for var in max_vars:
    cdo_all_seas(control_name, var, run, mean=False)
    cdo_all_seas(feedback_name, var, run, mean=True)

#end