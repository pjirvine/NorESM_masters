
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

# control 004-020 missing for old varialbes.
variables = ['QFLX'] #['LHFLX','QFLX','PRECC','PRECL','PRECTMX','TREFHT','TREFHTMX']

### List of RUNS groups ###

feedback_runs = [str(IDX+1).zfill(3) for IDX in xrange(21)]
control_runs = ['001','002','003','021']
# control short runs are all between 001-021 not included in control_runs
control_short_runs = [x for x in feedback_runs if x not in control_runs]

### search and dates for feedback / control

seach_format_feedback = "feedback.{run}.cam.h0.{var}."
dates_feedback = '202001-209912'

seach_format_control = "control.{run}.cam.h0.{var}."
dates_control = '201001-209912'

### merge years function

def merge_years_loop(seach_format, dates, runs, variables):

    for var in variables:
        for run in runs:

            run_name = seach_format.format(run=run, var=var)
            years_out = merge_years(run_name, dates, raw_monthly_dir, temp_dir)

            print run_name, years_out
#enddef

### Main merge functions ###

print "Feedback merge years"
merge_years_loop(seach_format_feedback, dates_feedback, feedback_runs, variables)
print "Control long merge years"
merge_years_loop(seach_format_control, dates_control, control_runs, variables)
print "Control short merge years"
merge_years_loop(seach_format_control, dates_control, control_short_runs, variables)

#end
