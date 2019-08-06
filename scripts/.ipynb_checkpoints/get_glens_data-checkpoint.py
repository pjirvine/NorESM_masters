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
Function to get all masks and weights
"""
def get_glens_masks_weights():
    """
    Generates all masks and weights needed for plots
    MASKS:
    'land_mask' - binary land mask where land fraction > 50%
    'land_noice_mask' - binary land mask without Greenland or Antarctica and where land fraction > 50%
    WEIGHTS:
    'pop' - gridcell weighting by population fraction
    'ag' - gridcell weighting by agricultural land fraction
    'area' - simple gridcell weighting by area
    'land_area' - land area weighting using raw land area fraction (not mask)
    'land_noice_area' - land area without Greenland and Antarctica weighting using raw land area fraction (not mask)
    """
    
    glen_dir = '/n/home03/pjirvine/keithfs1_pji/GLENS/fix/'
    
    """
    Masks
    """
    
    masks_weights = {}

    # turn field into array then squeeze off degenerate dimensions

    # land_noice mask
    land_ga_file = 'CCSM4_land_no_gr_ant.nc'
    f = Dataset(glen_dir + land_ga_file).variables['sftlf'][:].squeeze()
    land_noice_data = f
    masks_weights['land_noice_mask'] = land_noice_data > 0.5

    # land mask
    land_file = 'sftlf_CCSM4.nc'
    f = Dataset(glen_dir + land_file).variables['sftlf'][:].squeeze()
    land_data = f
    masks_weights['land_mask'] = land_data > 0.5

    """
    Weights
    """

    # pop weight
    pop_file = 'CCSM4_pop.nc'
    f = Dataset(glen_dir + pop_file).variables['pop'][:].squeeze()
    pop_data = f
    masks_weights['pop'] = pop_data / np.sum(pop_data)

    # ag weight
    ag_file = 'CCSM4_agriculture.nc'
    f = Dataset(glen_dir + ag_file).variables['fraction'][:].squeeze()
    ag_data = f
    masks_weights['ag'] = ag_data / np.sum(ag_data)
 
    # area weight
    weight_file = 'CCSM4_gridweights.nc'

    # get area weight, turn to array, squeeze off extra dims
    f = Dataset(glen_dir + weight_file).variables['cell_weights'][:].squeeze()
    weight_data = f
    masks_weights['area'] = weight_data # sums to 1.0

    # land area weight
    temp_data = land_data * weight_data
    masks_weights['land_area'] = temp_data / np.sum(temp_data)

    # land_noice area weight
    temp_data = land_noice_data * weight_data
    masks_weights['land_noice_area'] = temp_data / np.sum(temp_data)
    
    # 'land_mask', 'land_noice_mask'
    # 'pop', 'ag', 'area', 'land_area', 'land_noice_area'
    return masks_weights

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
    
    # get data
    glens_fileloc = glens_dir + glens_filename
    f = Dataset(glens_fileloc).variables[var][:]
    
    # convert units from K to C and from M/s to mm/day
    second_to_day = 60. * 60. * 24.
    var_mult = {'TREFHT':1.0, 'TREFHTMX':1.0, 'P-E':second_to_day * 1000., 'PRECTMX':second_to_day * 1000.,}
    var_const = {'TREFHT':-273.15, 'TREFHTMX':-273.15, 'P-E':0.0, 'PRECTMX':0.0,}
    data = var_mult[var] * f + var_const[var]
    
    # Return array but transpose order of dimensions from (t, y, x) to (x, y, t) and remove empty dimensions.
    return np.transpose(data.squeeze())
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
    'Half-GLENS'   - Scaled Half-GLENS  @ 2075-2094
    'Baseline-2'   - Shifted Half-GLENS @ 2075-2094  W/ alternate runs
    'RCP8.5-2'       - !!! NO ALTERNATE RUNS - NOT POSSIBLE !!!!
    'Full-GLENS-2'   - GLENS  @ 2075-2094  W/ alternate runs
    'Half-GLENS-2'   - Scaled Half-GLENS  @ 2075-2094  W/ alternate runs
    'RCP8.5-2'       - !!! NO ALTERNATE RUNS - NOT POSSIBLE !!!!
    ### NOT DONE ### 'Half-GLENS-time' - RCP8.5 @ 2010-2029 W/ alternate runs
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
    
    # BASELINE - RCP8.5 @ 2010-2029
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
    # Half-GLENS = RCP8.5 + 0.5*(GLENS-RCP8.5) @ 2075-2094
    elif case == 'Half-GLENS':
        # Get Full-GLENS data
        full_data_GLENS = [get_glens_annual(var, 'feedback', IDX, feedback_file_years) for IDX in core_runs]
        reduced_data_GLENS = [ IDX[:,:,t_index_feedback] for IDX in full_data_GLENS ]
        GLENS_stats = ensemble_stats(reduced_data_GLENS)
        # Get RCP8.5 data
        full_data_RCP85 = [get_glens_annual(var, 'control', IDX, control_file_years) for IDX in core_runs]
        reduced_data_RCP85 = [ IDX[:,:,t_index_feedback] for IDX in full_data_RCP85 ]
        RCP85_stats = ensemble_stats(reduced_data_RCP85)

        # Half-GLENS stats
        HalfGLENS_mean = RCP85_stats[0] + 0.5 * (GLENS_stats[0] - RCP85_stats[0])
        HalfGLENS_std = RCP85_stats[1] + 0.5 * (GLENS_stats[1] - RCP85_stats[1])
        
        return HalfGLENS_mean, HalfGLENS_std
    # BASELINE-2 - RCP8.5 @ 2010-2029 W/ alternate runs
    if case == 'Baseline-2':
        full_data = [get_glens_annual(var, 'control', IDX, control_short_file_years) for IDX in alt_runs]
        reduced_data = [ IDX[:,:,t_index_baseline] for IDX in full_data ]
        return ensemble_stats(reduced_data)
    #
    #### NO ALTERNATE RUNS AVAILABLE FOR RCP8.5 @ 2075-2094
    #
    # GLENS -2 @ 2075-2094 W/ alternate runs
    elif case == 'Full-GLENS-2':
        full_data = [get_glens_annual(var, 'feedback', IDX, feedback_file_years) for IDX in alt_runs]
        reduced_data = [ IDX[:,:,t_index_feedback] for IDX in full_data ]
        return ensemble_stats(reduced_data)
    # Half-GLENS -2 = RCP8.5 + 0.5*(GLENS-RCP8.5) @ 2075-2094  W/ alternate runs
    elif case == 'Half-GLENS-2':
        # Get Full-GLENS data
        full_data_GLENS = [get_glens_annual(var, 'feedback', IDX, feedback_file_years) for IDX in alt_runs]
        reduced_data_GLENS = [ IDX[:,:,t_index_feedback] for IDX in full_data_GLENS ]
        GLENS_stats = ensemble_stats(reduced_data_GLENS)
        # Get RCP8.5 data
        full_data_RCP85 = [get_glens_annual(var, 'control', IDX, control_file_years) for IDX in core_runs]
        reduced_data_RCP85 = [ IDX[:,:,t_index_feedback] for IDX in full_data_RCP85 ]
        RCP85_stats = ensemble_stats(reduced_data_RCP85)

        # Half-GLENS stats
        HalfGLENS_mean = RCP85_stats[0] + 0.5 * (GLENS_stats[0] - RCP85_stats[0])
        HalfGLENS_std = RCP85_stats[1] + 0.5 * (GLENS_stats[1] - RCP85_stats[1])
        
        return HalfGLENS_mean, HalfGLENS_std
    else:
        print(case, ' not listed')
        return None
# end def

"""
Generate means and stds for all variables and cases
"""

def get_all_cases_vars():
    
    vars_glens = ['TREFHT','TREFHTMX','P-E','PRECTMX']
    cases = ['Baseline','RCP8.5','Full-GLENS','Half-GLENS','Baseline-2','Full-GLENS-2','Half-GLENS-2'] # MORE TO ADD LATER
    all_data = {}
    for var in vars_glens:
        for case in cases:
            all_data[var,case] = ensemble_process(var,case)
    #endfor
    
    return all_data
#end def
