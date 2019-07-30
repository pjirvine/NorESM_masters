"""
List of functions used in the GLENS processing
- merge_years() - merges raw netcdf files together
- PE_makwe() - Makes PRECT and P-E monthly files from input monthly files
- ann_stats() - Calculates annual stats for key variables
- time_mean_std() - This calculated means and standard deviations over exp and baseline periods.

- linux_out() - runs a linux command and returns output
- sbatch_maker() - creates an SBATCH script
"""

from cdo import *
cdo = Cdo()
import my_shell_tools
import os
import os.path
import CMIP5_functions as cmip5

def merge_years(run_name, append, in_dir, out_dir):
    # This function finds all files that match the search and merges them.
    # If there is only 1 file it is copied to output directory.
    # run_data defines everything but year in filename.
    
    # FIRST - search for and return all matching netcdf files
    # search command that is executed in linux
    search_command = 'ls {directory}{search}'.format(directory=in_dir,search=run_name+'*.nc')
    # execute search command and store as temp1 string
    temp1=linux_out(search_command)
    # replace new lines ("\n") with spaces.
    temp2 = temp1.replace("\n"," ")
    
    cdo.mergetime(input=temp2, output=out_dir + run_name + append)
    
    return cdo.showyear(input=out_dir + run_name + append)



def PE_maker(run_name, in_dir, out_dir):
    """
    Makes PRECT and P-E monthly files out of PRECC, PRECL and LHFLX input files.
    
    !!! NOTE !!!
    run_name must include {var}, e.g. be of form: "feedback.021.cam.h0.{var}.202001-209912.nc"
    !!!      !!!
    """
    
    """
    make PRECT
    """
    
    file_1 = in_dir + run_name.format(var='PRECC')
    file_2 = in_dir + run_name.format(var='PRECL')

    cdo.add(input=file_1 + " " + file_2, output='temp.nc')
    cdo.chname('PRECC','PRECT', input = 'temp.nc', output= out_dir + run_name.format(var='PRECT'))
    
    """
    Evap = LHFLX * C
    """

    file_1 = in_dir + run_name.format(var='LHFLX')

    # convert Wm-2 to M/s using latent heat of vaporization
    lhflx_prect = 1 / (2.26*10**9)

    cdo.mulc(lhflx_prect, input=file_1, output='temp_evap.nc')

    """
    P-E = PRECT - EVAP
    """

    file_1 = out_dir + run_name.format(var='PRECT')
    file_2 = 'temp_evap.nc'

    cdo.sub(input = file_1 + ' ' + file_2, output = 'temp_pe.nc')
    cdo.chname('PRECT','P-E', input = 'temp_pe.nc', output= out_dir + run_name.format(var='P-E'))

    
def ann_stats(run_name, in_dir, out_dir):
    """
    Makes PRECT and P-E monthly files out of PRECC, PRECL and LHFLX input files.
    
    !!! NOTE !!!
    run_name must include {var}, e.g. be of form: "feedback.021.cam.h0.{var}.202001-209912.nc"
    !!!      !!!
    
    will output file with name of form: "feedback.021.cam.h0.{var}.ann.202001-209912.nc"
    """
    
    var = 'P-E'    # P-E - ann mean
    cdo.yearmonmean(input = in_dir + run_name.format(var=var), output = out_dir + run_name.format(var=var+'.ann'))
    var = 'TREFHT'    # TREFHT - ann mean
    cdo.yearmonmean(input = in_dir + run_name.format(var=var), output = out_dir + run_name.format(var=var+'.ann'))
    var = 'PRECTMX'    # PRECTMX - ann max
    cdo.yearmax(input = in_dir + run_name.format(var=var), output = out_dir + run_name.format(var=var+'.ann'))
    var = 'TREFHTMX'    # TREFHTMX - ann max
    cdo.yearmax(input = in_dir + run_name.format(var=var), output = out_dir + run_name.format(var=var+'.ann'))
    
#end def

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
#end def
    
def linux_out(command):
    import os
    # command = string as typed into linux terminal
    stream = os.popen(command)
    # return output
    return stream.read()


def sbatch_maker(script_command, sbatch_filename, more_commands='', JOB='Test', PART='general', NODES='1', CORES='1', MEM='100', TIME='0-00:10'):
    """
This function makes an SBATCH script called "sbatch_filename" which runs the script given in script_command.
The properties of the job will be set by the key words.
REMEMBER - This sbatch job will inherit
"""
    
    slurm_template = """#!/bin/bash
#
#SBATCH -J {JOB}
#SBATCH -p {PART}                # partition (queue)
#SBATCH -N {NODES}                      # number of nodes
#SBATCH -n {CORES}                      # number of cores
#SBATCH --mem={MEM}                 # memory pool for all cores
#SBATCH -t {TIME}                 # time (D-HH:MM)
#SBATCH -o /n/home03/pjirvine/slurm/out/SLURM.%j.%N.out        # STDOUT
#SBATCH -e /n/home03/pjirvine/slurm/err/SLURM.%j.%N.err        # STDERR
#SBATCH --mail-type=ALL
#                               Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=p.j.irvine@gmail.com
#                               Email to which notifications will be sent

{MORE_COMMANDS}
{SCRIPT}

############################## slurm ends ##############################
"""

    # Fill in the slurm template using kwargs
    sbatch_text = slurm_template.format(SCRIPT=script_command, MORE_COMMANDS=more_commands, JOB=JOB, PART=PART, NODES=NODES, CORES=CORES, MEM=MEM, TIME=TIME)

    # Write to file given by sbatch_filename
    with open(sbatch_filename, "w") as text_file:
        text_file.write(sbatch_text)
    
    # Return contents of sbatch file
    return sbatch_text

#def sbatch_maker ENDS
