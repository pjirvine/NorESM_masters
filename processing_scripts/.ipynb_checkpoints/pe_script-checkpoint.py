"""
PRECT and P-E monthly file maker script.
- Takes PRECC + PRECL to make PRECT
- and Takes PRECT and LHFLX to make P-E
"""

"""
Import modules
"""

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

feedback_runs = [str(IDX+1).zfill(3) for IDX in xrange(21)]
control_runs = ['001','002','003','021']
# control short runs are all between 001-021 not included in control_runs
control_short_runs = [x for x in feedback_runs if x not in control_runs]

feedback_name = "feedback.{run}.cam.h0.{var}.202001-209912.nc"
control_name_long = "control.{run}.cam.h0.{var}.201001-209912.nc"
control_name_short = "control.{run}.cam.h0.{var}.201001-203012.nc"

in_dir = "/n/home03/pjirvine/keithfs1_pji/GLENS/temp_merged/"
out_dir = "/n/home03/pjirvine/keithfs1_pji/GLENS/temp_p_pe/"

"""
Loop over control runs
"""

for run in control_runs:
        
    # Specifying run name but leaving {var} to be resolved in function
    run_name = control_name_long.format(run=run, var='{var}')
    PE_maker(run_name, in_dir, out_dir)

    print run_name
    
for run in control_short_runs:
        
    # Specifying run name but leaving {var} to be resolved in function
    run_name = control_name_short.format(run=run, var='{var}')
    PE_maker(run_name, in_dir, out_dir)

    print run_name
    
for run in feedback_runs:
    
    # Specifying run name but leaving {var} to be resolved in function
    run_name = feedback_name.format(run=run, var='{var}')
    PE_maker(run_name, in_dir, out_dir)

    print run_name