"""
This package 
"""


# import cf
from netCDF4 import Dataset
import numpy as np
from analysis import *
from config import *

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
def get_masks_weights():
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
    
    # Function to open netcdf and return squeezed data
    def get_data(filename, variable, file_dir='../NorESM_fix/'):
        f = Dataset(file_dir + filename).variables[variable][:].squeeze()
        return f
    
    """
    Masks
    """
    
    masks_weights = {}

    # turn field into array then squeeze off degenerate dimensions

    # land_noice mask and frac
    filename = 'NorESM1-M_land_no_gr_ant.nc'
    variable = 'sftlf'
    data = get_data(filename, variable)
    masks_weights['land_noice_mask'] = np.transpose(data > 0.5)
    masks_weights['land_noice_frac'] = np.transpose(data)

    # land mask and land frac
    filename = 'sftlf_fx_NorESM1-M_piControl_r0i0p0.nc'
    variable = 'sftlf'
    data = get_data(filename, variable)
    masks_weights['land_mask'] = np.transpose(data > 0.5)
    masks_weights['land_frac'] = np.transpose(data)
    
    """
    Weights
    """

    # pop weight
    filename = 'NorESM1-M_pop.nc'
    variable = 'pop'
    data = get_data(filename, variable)
    masks_weights['pop'] = np.transpose(data)

    # ag weight
    filename = 'NorESM1-M_agriculture.nc'
    variable = 'fraction'
    data = get_data(filename, variable)
    masks_weights['ag'] = np.transpose(data / np.sum(data))
 
    # area weight
    filename = 'NorESM1-M_weights.nc'
    variable = 'cell_weights'
    data = get_data(filename, variable)
    masks_weights['area'] = np.transpose(data)

    # land area weight
    filename = 'sftlf_fx_NorESM1-M_piControl_r0i0p0.nc'
    variable = 'sftlf'
    land_data = get_data(filename, variable)
    
    filename = 'NorESM1-M_weights.nc'
    variable = 'cell_weights'
    weight_data = get_data(filename, variable)
    
    data = land_data * weight_data
    masks_weights['land_area'] = np.transpose(data / np.sum(data))

    # land_noice area weight  
    filename = 'NorESM1-M_land_no_gr_ant.nc'
    variable = 'sftlf'
    land_noice_data = get_data(filename, variable)
    
    data = land_noice_data * weight_data
    masks_weights['land_noice_area'] = np.transpose(data / np.sum(data))
    
    # 'land_mask', 'land_noice_mask'
    # 'pop', 'ag', 'area', 'land_area', 'land_noice_area'
    return masks_weights

"""
Function to get the individual 3D annual series
"""
def get_NorESM_annual(var, exp, run):
    """
    Returns 3D netcdf data
    """    

    # use pr for varname for P-E
    if var == 'P-E':
        var_name = 'pr'
    else:
        var_name = var
    
    # Directory and filenames for annual timeseries of 2D data
    file_dir = '../NorESM_data/'
    filename = '{var}_Amon_NorESM1-ME_{exp}_{run}_2020-2100.nc'.format(exp=exp,run=run,var=var)
    
    # get data
    fileloc = file_dir + filename
    f = Dataset(fileloc).variables[var_name][:]
    
    # convert units from K to C and from M/s to mm/day
    second_to_day = 60. * 60. * 24.
    var_mult = {'tas':1.0, 'P-E':second_to_day, 'pr':second_to_day, 'evspsbl':second_to_day,}
    var_const = {'tas':-273.15, 'P-E':0., 'pr':0., 'evspsbl':0.,}
    data = var_mult[var] * f + var_const[var]
    
    # Return array but transpose order of dimensions from (t, y, x) to (x, y, t) and remove empty dimensions.
    return np.transpose(data.squeeze())
#end def

def ensemble_process(var, case):
    
    def ensemble_stats(ens_data, timeseries=False):
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
    # End def ensemble_stats
    
    # extract exp and t_index from case
    exp = case['exp']
    t_index = case['t_index']
    
    # Create a list of data from each run in this exp, var combo
    full_data = [get_NorESM_annual(var, exp, run) for run in runs]
    # Include only that data which falls in range of t_index, i.e. over year range specified
    reduced_data = [ IDX[:,:,t_index] for IDX in full_data ]
    # Calculate ensemble stats and return
    return ensemble_stats(reduced_data)

"""
Generate means and stds for all variables and cases
"""

def get_all_cases_vars_noresm():
    
    # The list of years corresponding to the array indices.
    year_idxs = np.array([IDX + 2020 for IDX in range(81)])
    
    #Specify time indexes for runs  
    t_index_baseline = np.where((year_idxs > 2019) & (year_idxs < 2040))[0] # this captures 2020 ... 2039
    t_index_exps = np.where((year_idxs > 2079) & (year_idxs < 2100))[0] # this captures 2080 ... 2099

    # Cases specify a combination of experiment and time-period
    cases = {'baseline':{'exp':'rcp45','t_index':t_index_baseline},
             'rcp45':{'exp':'rcp45','t_index':t_index_exps},
             'rcp85':{'exp':'rcp85','t_index':t_index_exps},
             'G6sulf':{'exp':'G6sulf','t_index':t_index_exps},
             'G6ss':{'exp':'G6ss','t_index':t_index_exps},
             'G6cct':{'exp':'G6cct','t_index':t_index_exps},
            }
    
    vars = ['tas','pr','evspsbl','P-E']

    all_data = {}
    for var in vars:
        for case_key, case_value in cases.items():
            all_data[var,case_key] = ensemble_process(var,case_value)
    #endfor
    
    return all_data
#end def

"""
Generate better / worse off and all anomalies for 3 cases
"""

def better_worse_full_data(all_data, case_sg, case_CO2, case_ctrl, var, weight, nyears=60, ttest_level=0.1, anom_type='standard'):
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