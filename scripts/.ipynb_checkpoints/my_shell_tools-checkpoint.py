"""
Python functions of use for interacting with the shell
"""

import subprocess
import os

"""
Define basic sub-routines:
"""

# This function grabs files from file_dir using the specified ls_search
def file_grabber(file_dir, ls_search):

    # run shell command to 'ls' contents of directory and store
    file_list_string = subprocess.check_output('ls '+file_dir+ls_search, stderr=subprocess.STDOUT, shell=True)
    file_list_string = file_list_string.replace(file_dir,'') # remove directory from name

    # convert string to list and remove the empty last entry
    file_list = file_list_string.split('\n') # split on new lines
    file_list = file_list[0:len(file_list)-1] # exclude last entry

    return file_list

def linux_out(command):
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

"""
FIN
"""

