"""
This package 
"""


# import cf
from netCDF4 import Dataset
import numpy as np

"""
Function to get lons, lats and weights from example netcdf file.
"""
def get_lons_lats_weights():

    #Get example netcdf
    glens_dir = '/n/home03/pjirvine/keithfs1_pji/GLENS/combined_annual_data/'
    glens_filename = 'control.001.cam.h0.TREFHT.ann.201001-209912.nc'
    glens_fileloc = glens_dir + glens_filename
    test_nc = Dataset(glens_fileloc)

    # produce lons, lats
    lons = np.array(test_nc.variables['lon'][:])
    lats = np.array(test_nc.variables['lat'][:])

    # get grid-weights by latitude
    gw = test_nc.variables['gw'][:]
    # repeat along lons dimension
    gw_2D = np.tile(gw, (lons.size,1))
    # normalize
    weights = gw_2D / np.sum(gw_2D)
    
    return lons, lats, weights

"""
Function to get the individual 3D annual series
"""
def get_glens_annual(var, exp, run, file_years):
    """
    Returns 3D netcdf data
    """
    
    # Directory and filenames for annual timeseries of 2D data
    glens_dir = '/n/home03/pjirvine/keithfs1_pji/GLENS/combined_annual_data/'
    glens_filename = '{exp}.{run}.cam.h0.{var}.ann.{years}.nc'.format(exp=exp,run=run,var=var,years=file_years)
    
    glens_fileloc = glens_dir + glens_filename
    f = Dataset(glens_fileloc).variables[var][:]
        
    # Return array but transpose order of dimensions from (t, y, x) to (x, y, t) and remove empty dimensions.
    return np.transpose(f.squeeze())
#end def

"""
Functions to process ensemble for given var and case
"""
def ensemble_process(var,case):
    """
    For the given case and var generate the ensemble mean and standard deviation results
    CASES:
    'Baseline'     - RCP8.5 @ 2010-2029
    'RCP8.5'       - RCP8.5 @ 2075-2094
    'Full-GLENS'   - GLENS  @ 2075-2094
    ### NOT DONE ### 'Half-GLENS'   - Scaled Half-GLENS  @ 2075-2094
    ### NOT DONE ### 'Baseline-2'   - Shifted Half-GLENS @ 2075-2094
    ### NOT DONE ### 'Half-GLENS-2' - RCP8.5 @ 2010-2029 W/ alternate runs
    """
    
    def ensemble_stats(ens_data):
        """
        Calculates the mean and standard deviation using all years in all ensemble members
        INPUT: ens_data = list[ output of GET_GLENS_ANNUAL() ] = list of data[lon,lat,time]
        OUTPUT: list[ ens_mean[lon,lat], ens_std[lon,lat] ]
        """

        ens_combined = np.concatenate(ens_data, axis=2)
        ens_mean = np.mean(ens_combined, axis=2)
        ens_std = np.std(ens_combined, axis=2)

        return ens_mean, ens_std
    #end def
    
    """
    Specify years of experiments and associated indices for annual files
    """

    years_control = np.array([IDX + 2010 for IDX in range(90)])
    years_feedback = np.array([IDX + 2020 for IDX in range(80)])

    #Generate the indices for the range of years in each case.
    # [0] added as a 2 element tuple with an array and an empty slot returned rather than an array
    t_index_control = np.where((years_control > 2074) & (years_control < 2095))[0]
    t_index_baseline = np.where((years_control > 2009) & (years_control < 2030))[0]
    t_index_feedback = np.where((years_feedback > 2074) & (years_feedback < 2095))[0]
    
    # year ranges which appears in filename
    control_file_years = '201001-209912'
    control_short_file_years = '201001-203012'
    feedback_file_years = '202001-209912'
    
    # The same 4 core runs will be used for the RCP8.5, baseline and GLENS data
    core_runs = ['001','002','003','021']
    # 4 alternative runs will be used to generate an alternative control case
    alt_runs = ['004','005','006','007']
    
    """
    Produce data according to rules for each case
    """
    
    # RCP8.5 @ 2010-2029
    if case == 'Baseline':
        full_data = [get_glens_annual(var, 'control', IDX, control_file_years) for IDX in core_runs]
        reduced_data = [ IDX[:,:,t_index_baseline] for IDX in full_data ]
        return ensemble_stats(reduced_data)
    # RCP8.5 @ 2075-2094
    elif case == 'RCP8.5':
        full_data = [get_glens_annual(var, 'control', IDX, control_file_years) for IDX in core_runs]
        reduced_data = [ IDX[:,:,t_index_control] for IDX in full_data ]
        return ensemble_stats(reduced_data)
    # GLENS @ 2075-2094
    elif case == 'Full-GLENS':
        full_data = [get_glens_annual(var, 'feedback', IDX, feedback_file_years) for IDX in core_runs]
        reduced_data = [ IDX[:,:,t_index_feedback] for IDX in full_data ]
        return ensemble_stats(reduced_data)
    else:
        print(case, ' not listed')
        return None
# end def

"""
Generate means and stds for all variables and cases
"""

def get_all_cases_vars():
    
    vars_glens = ['TREFHT','TREFHTMX','P-E','PRECTMX']
    cases = ['Baseline','RCP8.5','Full-GLENS'] # MORE TO ADD LATER
    all_data = {}
    for var in vars_glens:
        for case in cases:
            all_data[var,case] = ensemble_process(var,case)
    #endfor
    
    return all_data
#end def
