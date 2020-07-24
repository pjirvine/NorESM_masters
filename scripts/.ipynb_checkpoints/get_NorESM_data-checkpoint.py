"""
This package 
"""


# import cf
from netCDF4 import Dataset
import numpy as np
from analysis import *

"""
Function to get lons, lats and weights from example netcdf file.

Wei - you will need to change the 3 GLENS lines below to fit a NorESM file, it doesn't matter which.

"""
def get_lons_lats_weights():

    #Get example netcdf
    file_dir = '../NorESM_fix/'
    filename = 'NorESM1-M_weights.nc'
    fileloc = file_dir + filename
    test_nc = Dataset(fileloc)

    # produce lons, lats
    lons = np.array(test_nc.variables['lon'][:])
    lats = np.array(test_nc.variables['lat'][:])

    # get grid-weights by latitude
    gw = test_nc.variables['cell_weights'][:]
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
    
    glen_dir = '../glens_fix/'
    
    """
    Masks
    """
    
    masks_weights = {}

    # turn field into array then squeeze off degenerate dimensions

    # land_noice mask
    land_ga_file = 'CCSM4_land_no_gr_ant.nc'
    f = Dataset(glen_dir + land_ga_file).variables['sftlf'][:].squeeze()
    land_noice_data = f
    masks_weights['land_noice_mask'] = np.transpose(land_noice_data > 0.5)

    # land mask
    land_file = 'sftlf_CCSM4.nc'
    f = Dataset(glen_dir + land_file).variables['sftlf'][:].squeeze()
    land_data = f
    masks_weights['land_mask'] = np.transpose(land_data > 0.5)

    # land_noice fraction
    land_ga_file = 'CCSM4_land_no_gr_ant.nc'
    f = Dataset(glen_dir + land_ga_file).variables['sftlf'][:].squeeze()
    land_noice_data = f
    masks_weights['land_noice_frac'] = np.transpose(land_noice_data)

    # land mask fraction
    land_file = 'sftlf_CCSM4.nc'
    f = Dataset(glen_dir + land_file).variables['sftlf'][:].squeeze()
    land_data = f
    masks_weights['land_frac'] = np.transpose(land_data)
    
    """
    Weights
    """

    # pop weight
    pop_file = 'CCSM4_pop.nc'
    f = Dataset(glen_dir + pop_file).variables['pop'][:].squeeze()
    pop_data = f
    masks_weights['pop'] = np.transpose(pop_data / np.sum(pop_data))

    # ag weight
    ag_file = 'CCSM4_agriculture.nc'
    f = Dataset(glen_dir + ag_file).variables['fraction'][:].squeeze()
    ag_data = f
    masks_weights['ag'] = np.transpose(ag_data / np.sum(ag_data))
 
    # area weight
    weight_file = 'CCSM4_gridweights.nc'

    # get area weight, turn to array, squeeze off extra dims
    f = Dataset(glen_dir + weight_file).variables['cell_weights'][:].squeeze()
    weight_data = f
    masks_weights['area'] = np.transpose(weight_data) # sums to 1.0

    # land area weight
    temp_data = land_data * weight_data
    masks_weights['land_area'] = np.transpose(temp_data / np.sum(temp_data))

    # land_noice area weight
    temp_data = land_noice_data * weight_data
    masks_weights['land_noice_area'] = np.transpose(temp_data / np.sum(temp_data))
    
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
    glens_dir = '../glens_data/'
    glens_filename = '{exp}.{run}.cam.h0.{var}.ann.{years}.nc'.format(exp=exp,run=run,var=var,years=file_years)
    
    # get data
    glens_fileloc = glens_dir + glens_filename
    f = Dataset(glens_fileloc).variables[var][:]
    
    # convert units from K to C and from M/s to mm/day
    second_to_day = 60. * 60. * 24.
    var_mult = {'TREFHT':1.0, 'TREFHTMX':1.0, 'P-E':second_to_day * 1000., 'PRECTMX':second_to_day * 1000., 'PRECT':second_to_day * 1000.,}
    var_const = {'TREFHT':-273.15, 'TREFHTMX':-273.15, 'P-E':0.0, 'PRECTMX':0.0, 'PRECT':0.0}
    data = var_mult[var] * f + var_const[var]
    
    # Return array but transpose order of dimensions from (t, y, x) to (x, y, t) and remove empty dimensions.
    return np.transpose(data.squeeze())
#end def

"""
Functions to process ensemble for given var and case
"""
def ensemble_process(var,case, timeseries=False):
    """
    For the given case and var generate the ensemble mean and standard deviation results
    CASES:
    'Baseline'     - RCP8.5 @ 2010-2029
    'RCP8.5'       - RCP8.5 @ 2075-2094
    'Full-GLENS'   - GLENS  @ 2075-2094
    'Half-GLENS'   - Scaled Half-GLENS  @ 2075-2094
    'Half-GLENS-time'- Shifted Half-GLENS  @ 2044-2063
    'Baseline-2'   - RCP8.5 @ 2010-2029  W/ alternate runs
    'RCP8.5-2'       - !!! NO ALTERNATE RUNS - NOT POSSIBLE !!!!
    'Full-GLENS-2'   - GLENS  @ 2075-2094  W/ alternate runs
    'Half-GLENS-2'   - Scaled Half-GLENS  @ 2075-2094  W/ alternate runs
    """
    
    def ensemble_stats(ens_data, timeseries):
        """
        Calculates the mean and standard deviation using all years in all ensemble members
        INPUT: ens_data = list[ output of GET_GLENS_ANNUAL() ] = list of data[lon,lat,time]
        timeseries = False --> OUTPUT: list[ ens_mean[lon,lat], ens_std[lon,lat] ]
        timeseries = True --> OUTPUT: list[ ens_combined[lon,lat,time], ens_mean[lon,lat,time], ens_std[lon,lat,time] ]       
        """

        if timeseries:
            ens_combined = np.stack(ens_data, axis=3)
            ens_mean = np.mean(ens_combined, axis=3)
            ens_std = np.std(ens_combined, axis=3)
            return ens_mean, ens_std
        else: 
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
    # Years found using offline calculation with this function call: closest_years_to_frac_GLENS(0.5)
    t_index_feedback_half = np.where((years_feedback > 2043) & (years_feedback < 2064))[0]
    t_index_control_half = np.where((years_control > 2043) & (years_control < 2064))[0]
    
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
        return ensemble_stats(reduced_data, timeseries)
    # RCP8.5 @ 2075-2094
    elif case == 'RCP8.5':
        full_data = [get_glens_annual(var, 'control', IDX, control_file_years) for IDX in core_runs]
        reduced_data = [ IDX[:,:,t_index_control] for IDX in full_data ]
        return ensemble_stats(reduced_data, timeseries)
    # GLENS @ 2075-2094
    elif case == 'Full-GLENS':
        full_data = [get_glens_annual(var, 'feedback', IDX, feedback_file_years) for IDX in core_runs]
        reduced_data = [ IDX[:,:,t_index_feedback] for IDX in full_data ]
        return ensemble_stats(reduced_data, timeseries)
    # Half-GLENS = RCP8.5 + 0.5*(GLENS-RCP8.5) @ 2075-2094
    elif case == 'Half-GLENS':
        # Get Full-GLENS data
        full_data_GLENS = [get_glens_annual(var, 'feedback', IDX, feedback_file_years) for IDX in core_runs]
        reduced_data_GLENS = [ IDX[:,:,t_index_feedback] for IDX in full_data_GLENS ]
        GLENS_stats = ensemble_stats(reduced_data_GLENS, timeseries)
        # Get RCP8.5 data
        full_data_RCP85 = [get_glens_annual(var, 'control', IDX, control_file_years) for IDX in core_runs]
        reduced_data_RCP85 = [ IDX[:,:,t_index_control] for IDX in full_data_RCP85 ]
        RCP85_stats = ensemble_stats(reduced_data_RCP85, timeseries)

        # Half-GLENS stats
        HalfGLENS_mean = RCP85_stats[0] + 0.5 * (GLENS_stats[0] - RCP85_stats[0])
        HalfGLENS_std = RCP85_stats[1] + 0.5 * (GLENS_stats[1] - RCP85_stats[1])
        
        return HalfGLENS_mean, HalfGLENS_std
    # Half-GLENS-time = RCP8.5 @ 2075-2094 + (RCP8.5 - GLENS) @ 2044-2063
    elif case == 'Half-GLENS-time':
        # Get Full-GLENS data @ 2044-2063
        full_data_GLENS = [get_glens_annual(var, 'feedback', IDX, feedback_file_years) for IDX in core_runs]
        reduced_data_GLENS_44_63 = [ IDX[:,:,t_index_feedback_half] for IDX in full_data_GLENS ]
        GLENS_stats_44_63 = ensemble_stats(reduced_data_GLENS_44_63, timeseries)
        # Get RCP8.5 data @ 2044-2063
        full_data_RCP85 = [get_glens_annual(var, 'control', IDX, control_file_years) for IDX in core_runs]
        reduced_data_RCP85_44_63 = [ IDX[:,:,t_index_control_half] for IDX in full_data_RCP85 ]
        RCP85_stats_44_63 = ensemble_stats(reduced_data_RCP85_44_63, timeseries)
        # Get RCP8.5 data @ 2075-2094
        full_data_RCP85 = [get_glens_annual(var, 'control', IDX, control_file_years) for IDX in core_runs]
        reduced_data_RCP85 = [ IDX[:,:,t_index_control] for IDX in full_data_RCP85 ]
        RCP85_stats = ensemble_stats(reduced_data_RCP85, timeseries)
        
        # Half-GLENS stats
        HalfGLENS_time_mean = RCP85_stats[0] + (GLENS_stats_44_63[0] - RCP85_stats_44_63[0])
        HalfGLENS_time_std = RCP85_stats[1] + (GLENS_stats_44_63[1] - RCP85_stats_44_63[1])
        
        return HalfGLENS_time_mean, HalfGLENS_time_std
    # BASELINE-2 - RCP8.5 @ 2010-2029 W/ alternate runs
    if case == 'Baseline-2':
        full_data = [get_glens_annual(var, 'control', IDX, control_short_file_years) for IDX in alt_runs]
        reduced_data = [ IDX[:,:,t_index_baseline] for IDX in full_data ]
        return ensemble_stats(reduced_data, timeseries)
    #
    #### NO ALTERNATE RUNS AVAILABLE FOR RCP8.5 @ 2075-2094
    #
    # GLENS -2 @ 2075-2094 W/ alternate runs
    elif case == 'Full-GLENS-2':
        full_data = [get_glens_annual(var, 'feedback', IDX, feedback_file_years) for IDX in alt_runs]
        reduced_data = [ IDX[:,:,t_index_feedback] for IDX in full_data ]
        return ensemble_stats(reduced_data, timeseries)
    # Half-GLENS -2 = RCP8.5 + 0.5*(GLENS-RCP8.5) @ 2075-2094  W/ alternate runs
    elif case == 'Half-GLENS-2':
        # Get Full-GLENS data
        full_data_GLENS = [get_glens_annual(var, 'feedback', IDX, feedback_file_years) for IDX in alt_runs]
        reduced_data_GLENS = [ IDX[:,:,t_index_feedback] for IDX in full_data_GLENS ]
        GLENS_stats = ensemble_stats(reduced_data_GLENS, timeseries)
        # Get RCP8.5 data
        full_data_RCP85 = [get_glens_annual(var, 'control', IDX, control_file_years) for IDX in core_runs]
        reduced_data_RCP85 = [ IDX[:,:,t_index_feedback] for IDX in full_data_RCP85 ]
        RCP85_stats = ensemble_stats(reduced_data_RCP85, timeseries)

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
    
    vars_glens = ['TREFHT','TREFHTMX','P-E','PRECTMX','PRECT']
    cases = ['Baseline','RCP8.5','Full-GLENS','Half-GLENS','Half-GLENS-time','Baseline-2','Full-GLENS-2','Half-GLENS-2'] # MORE TO ADD LATER
    all_data = {}
    for var in vars_glens:
        for case in cases:
            all_data[var,case] = ensemble_process(var,case)
    #endfor
    
    return all_data
#end def

"""
Generate better / worse off and all anomalies for 3 cases
"""

def better_worse_full_data(all_data, case_sg, case_CO2, case_ctrl, var, weight, nyears=80, ttest_level=0.1, anom_type='standard'):
    """
    Given 3 cases, a variable and a weight --> returns dictionary of better, worse off etc.
    """
    
    # Get means and stds for each case
    case_sg_mean, case_sg_std = all_data[(var, case_sg)]
    case_CO2_mean, case_CO2_std = all_data[(var, case_CO2)]
    case_ctrl_mean, case_ctrl_std = all_data[(var, case_ctrl)]
    
    # Returns better[], worse[], dont_know[]
    better, worse, dont_know = better_worse_off(case_sg_mean, case_sg_std, case_CO2_mean, case_CO2_std, case_ctrl_mean, case_ctrl_std, nyears, ttest_level)   
    certain = better + worse
    
    # calculate which anom is greater 
    CO2_anom = case_CO2_mean - case_ctrl_mean
    sg_anom = case_sg_mean - case_ctrl_mean
    sg_CO2_anom = case_sg_mean - case_CO2_mean
    b_nosign = abs(sg_anom) < abs(CO2_anom)
    w_nosign = abs(sg_anom) >= abs(CO2_anom)
    
    # Modify output anom_type
    if anom_type == 'standard':
        pass
    # % change anomalies
    elif anom_type == 'pc':
        CO2_anom = 100. * ((case_CO2_mean / case_ctrl_mean) - 1.0)
        sg_anom = 100. * ((case_sg_mean / case_ctrl_mean) - 1.0)
        sg_CO2_anom = 100. * ((case_sg_mean / case_CO2_mean) - 1.0)
    # anomalies in terms of control STDS
    elif anom_type == 'SD':
        CO2_anom = (case_CO2_mean - case_ctrl_mean) / case_ctrl_std
        sg_anom = (case_sg_mean - case_ctrl_mean) / case_ctrl_std
        sg_CO2_anom = (case_sg_mean - case_CO2_mean) / case_ctrl_std # Note this is CTRL STDs
    else:
        print('not recognized anom type: ' + anom_type)
        return None
    
    # calculate whether CO2 anom and sg-CO2 anom are significant
    CO2_sign = ttest_sub(case_CO2_mean, case_CO2_std, nyears, case_ctrl_mean, case_ctrl_std, nyears) < ttest_level
    sg_sign = ttest_sub(case_sg_mean, case_sg_std, nyears, case_ctrl_mean, case_ctrl_std, nyears) < ttest_level
    sg_CO2_sign = ttest_sub(case_sg_mean, case_sg_std, nyears, case_CO2_mean, case_CO2_std, nyears) < ttest_level
    
    # create dictionary of masks (true/false) for each outcome
    masks = {}
    masks['better'] = better.flatten()
    masks['worse'] = worse.flatten()
    masks['dont_know'] = dont_know.flatten()
    masks['certain'] = certain.flatten()
    masks['b_nosign'] = b_nosign.flatten()
    masks['w_nosign'] = w_nosign.flatten()
    masks['CO2_sign'] = CO2_sign.flatten()
    masks['sg_sign'] = sg_sign.flatten()
    masks['sg_CO2_sign'] = sg_CO2_sign.flatten()
    
    # Create dictionary of masked weights for each outcome
    weights = {}
       
    def weight_func(mask, weight):
        return weight.flatten() * mask.flatten()
    
    for key, value in masks.items():
        weights[key] = weight_func(value,weight)
          
    # Create dictionary of weighted fraction for each mask
    fractions = {} 
    
    def weighted_frac(weight, weight_orig):
        fraction = np.sum(weight) / np.sum(weight_orig)
        if fraction > 1:
            print('weight problem', fraction)
        return fraction
    
    for key, value in weights.items():
        fractions[key] = weighted_frac(value,weight)
        
    # return all 3 anomalies, and masks, weights and fractions for each better / worse off outcome
    return sg_anom, CO2_anom, sg_CO2_anom, masks, weights, fractions
#end def
