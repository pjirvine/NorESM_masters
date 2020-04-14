
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

# RUN NUMBERS
feedback_runs = [str(IDX+1).zfill(3) for IDX in xrange(21)]
control_runs = ['001','002','003','021']
# control short runs are all between 001-021 not included in control_runs
control_short_runs = [x for x in feedback_runs if x not in control_runs]

# VARIABLES
variables = ['TREFHT','TREFHTMX','PRECTMX','P-E','PRECT','QFLX']

# DATES
date_feedback = '202001-209912'
date_control_long = '201001-209912'
date_control_short = '201001-203012'

date_baseline = '2010-2029'
date_exp = '2075-2094'

# FILE NAMES
feedback_name = "feedback.{run}.cam.h0.{var}.ann.{date}.{append}" # append = 'nc' or 'annmean.nc'
control_name_long = "control.{run}.cam.h0.{var}.ann.{date}.{append}"
control_name_short = "control.{run}.cam.h0.{var}.ann.{date}.{append}"

# directories
in_dir = "/n/home03/pjirvine/keithfs1_pji/GLENS/combined_annual_data/"
out_dir = "/n/home03/pjirvine/keithfs1_pji/GLENS/processed/"

# YEARS
# generate a list of years and convert to string with comma delimiter
baseline_years = ",".join([str(2010 + IDX) for IDX in xrange(20)])
exp_years = ",".join([str(2075 + IDX) for IDX in xrange(20)])

# TIME_MEAN_STD FUNCTION
#
def time_mean_std(name, years, run, var, date_in, date_out):
    """
    This function calculates the means and standard deviations
    """
    # name has to be in this format: "feedback.{run}.cam.h0.{var}.{date}.{append}"

    # Time stat function
    def timestat(in_dir, in_name, out_dir, out_name, years, stat):
        cdo.selyear(years, input=in_dir+in_name, output='temp.nc')
        if stat == 'mean':
            cdo.timmean(input='temp.nc', output=out_dir+out_name)
        elif stat == 'std':
            cdo.timstd(input='temp.nc', output=out_dir+out_name)
        else:
            return "enter mean or std for stat"

    # Format input and output names
    in_name = name.format(run=run,var=var,date=date_in,append='nc')
    out_name_mean = name.format(run=run,var=var,date=date_out,append='annmean.nc')
    out_name_std = name.format(run=run,var=var,date=date_out,append='annstd.nc')

    # Calculate mean and STD
    timestat(in_dir, in_name, out_dir, out_name_mean, years, 'mean')
    timestat(in_dir, in_name, out_dir, out_name_std, years, 'std')
#end def

"""
Start main loops
"""

for var in variables:

    # Control long - for EXP and baseline
    for run in control_runs:

        time_mean_std(control_name_long, exp_years, run, var, date_control_long, date_exp)
        time_mean_std(control_name_long, baseline_years, run, var, date_control_long, date_baseline)


    # Control short - for basline
    for run in control_short_runs:

        time_mean_std(control_name_short, baseline_years, run, var, date_control_short, date_baseline)

    # Feedback - for exp
    for run in feedback_runs:

        time_mean_std(feedback_name, exp_years, run, var, date_feedback, date_exp)

#end var
