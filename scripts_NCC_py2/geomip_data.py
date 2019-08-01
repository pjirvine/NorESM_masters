"""
This python script contains variables and functions that are required to gather up / generate the data
for the fraction better off project.
These functions will mostly be specific to this project.
"""

import numpy as np
import os.path
# import cf
from netCDF4 import Dataset
from scipy.stats import ttest_ind_from_stats
from copy import copy
from cdo import *
cdo = Cdo()

from analysis import *

"""
Define list of variables and their properties
"""

var_list = ['tas_Amon','txxETCCDI_yr','pr_Amon','rx5dayETCCDI_yr','p-e_Amon']
var_shortname = {'tas_Amon':'tas','txxETCCDI_yr':'txx','pr_Amon':'pr','rx5dayETCCDI_yr':'rx5day','p-e_Amon':'p-e'}
var_longname = {'tas_Amon':'2m Temperature','txxETCCDI_yr':'Max Temperature','pr_Amon':'Precipitation','rx5dayETCCDI_yr':'5-Day Max Precip.','p-e_Amon':'Precip. - Evap.'}
var_offsets = {'tas_Amon':-273.15, 'pr_Amon':0.0, 'rx5dayETCCDI_yr':0.0, 'txxETCCDI_yr':0.0, 'p-e_Amon':0.0}
var_mults = {'tas_Amon':1.0, 'pr_Amon':60.0*60.0*24.0, 'rx5dayETCCDI_yr':1.0, 'txxETCCDI_yr':1.0, 'p-e_Amon':60.0*60.0*24.0}
var_pcs = {'tas_Amon':False, 'pr_Amon':True, 'rx5dayETCCDI_yr':True, 'txxETCCDI_yr':False, 'p-e_Amon':True}
# all but NorESM1-M are regridded
var_name_mod = {'tas_Amon':"", 'pr_Amon':"", 'rx5dayETCCDI_yr':"_144x96", 'txxETCCDI_yr':"_144x96", 'p-e_Amon':""}

"""
Model - exp - runs + notes
"""

model_exp_runs = {}
model_exp_runs['BNU-ESM_piControl'] = ['r1i1p1'] # Issue! G1 r1 to be converted to r2
model_exp_runs['CanESM2_piControl'] = ['r1i1p1']
model_exp_runs['CCSM4_piControl'] = ['r2i1p1','r4i1p1'] # Issue! Waiting on separate to do
model_exp_runs['CESM-CAM5.1-FV_piControl'] = ['r1i1p1']
model_exp_runs['CSIRO-Mk3L-1-2_piControl'] = ['r1i1p1']
model_exp_runs['EC-EARTH_piControl'] = ['r1i1p1'] 
model_exp_runs['GISS-E2-R_piControl'] = ['r2i1p1','r1i1p1','r3i1p1','r1i1p2','r1i1p2'] # Issue! parts 2 and 3 in picontrol?
model_exp_runs['HadCM3_piControl'] = ['r1i1p1']
model_exp_runs['HadGEM2-ES_piControl'] = ['r1i1p1']
model_exp_runs['IPSL-CM5A-LR_piControl'] = ['r1i1p1']
model_exp_runs['MIROC-ESM_piControl'] = ['r1i1p1']
model_exp_runs['MPI-ESM-LR_piControl'] = ['r1i1p1']
model_exp_runs['NorESM1-M_piControl'] = ['r1i1p1']

model_exp_runs['BNU-ESM_abrupt4xCO2'] = ['r1i1p1'] # Issue! G1 r1 to be converted to r2
model_exp_runs['CanESM2_abrupt4xCO2'] = ['r1i1p1']
model_exp_runs['CCSM4_abrupt4xCO2'] = ['r1i1p1','r3i1p1'] # Issue! Waiting on separate to do
model_exp_runs['CESM-CAM5.1-FV_abrupt4xCO2'] = ['r1i1p1','r3i1p1']
model_exp_runs['CSIRO-Mk3L-1-2_abrupt4xCO2'] = ['r1i1p1','r2i1p1','r3i1p1']
model_exp_runs['EC-EARTH_abrupt4xCO2'] = ['r1i1p1'] 
model_exp_runs['GISS-E2-R_abrupt4xCO2'] = ['r2i1p1','r1i1p1','r3i1p1'] # Issue! parts 2 and 3 in picontrol?
model_exp_runs['HadCM3_abrupt4xCO2'] = ['r1i1p1','r2i1p1','r3i1p1']
model_exp_runs['HadGEM2-ES_abrupt4xCO2'] = ['r1i1p1']
model_exp_runs['IPSL-CM5A-LR_abrupt4xCO2'] = ['r1i1p1']
model_exp_runs['MIROC-ESM_abrupt4xCO2'] = ['r1i1p1']
model_exp_runs['MPI-ESM-LR_abrupt4xCO2'] = ['r1i1p1']
model_exp_runs['NorESM1-M_abrupt4xCO2'] = ['r1i1p1']

model_exp_runs['BNU-ESM_G1'] = ['r2i1p1','r1i1p1'] # Issue! G1 r1 to be converted to r2
model_exp_runs['CanESM2_G1'] = ['r1i1p1','r2i1p1','r3i1p1']
model_exp_runs['CCSM4_G1'] = ['r1i1p1','r2i1p1'] # Issue! Waiting on separate to do
model_exp_runs['CESM-CAM5.1-FV_G1'] = ['r1i1p1']
model_exp_runs['CSIRO-Mk3L-1-2_G1'] = ['r1i1p1','r2i1p1','r3i1p1']
model_exp_runs['EC-EARTH_G1'] = ['r1i1p1'] 
model_exp_runs['GISS-E2-R_G1'] = ['r2i1p1','r1i1p1','r3i1p1'] # Issue! parts 2 and 3 in picontrol?
model_exp_runs['HadCM3_G1'] = ['r1i1p1','r2i1p1','r3i1p1']
model_exp_runs['HadGEM2-ES_G1'] = ['r1i1p1']
model_exp_runs['IPSL-CM5A-LR_G1'] = ['r1i1p1']
model_exp_runs['MIROC-ESM_G1'] = ['r1i1p1']
model_exp_runs['MPI-ESM-LR_G1'] = ['r2i1p1']
model_exp_runs['NorESM1-M_G1'] = ['r1i1p1']

"""
###
Get GeoMIP Masks
###
"""

def get_masks_weights(model_in, variable=""): 
    
    """
    This function gathers up all the masks and weights listed below and returns them.
    
    If variable is Not None:
        It checks if the variable is an extreme and if so uses the NorESM1-M grid as
        all extreme variables are on this grid.
        
    If no original resolution ice is available the regridded NorESM1-M ice is used instead.
    
    Return:
        All masks, etc. are from 0 - 100% in input files but all are output as 0-1
        All weights total to 1 on output.
    """
    
    if "ETCCDI" in variable: # Use NorESM1-M if  variable is an extreme one
        model = "NorESM1-M"
    else:
        model = model_in
    
    # Define directory format
    in_dir_base = "/n/home03/pjirvine/keithfs1_pji/geomip_archive/final_data/{model}/fix/"
    in_dir = in_dir_base.format(model = model)

    # Define file names for land and ice
    nc_file_base = "{fx_type}_{model}{append}.nc"
    land_file = nc_file_base.format(model = model, fx_type = 'sftlf', append='')
        
    # Check if land file is present
    if not os.path.isfile(in_dir+land_file):
        return "No land mask file"
    
    # Load numpy array from land file
    land = Dataset(in_dir + land_file).variables['sftlf'][:]
    
    land_noice_file = "{model}_land_no_gr_ant.nc".format(model = model)
    # Load numpy array
    land_noice = Dataset(in_dir + land_noice_file).variables['sftlf'][:]
    
    """
    Get weights
    """
    
    # calculate grid_weights
    # Check if gridweights file is present (this is needed as some mask files don't contain gridweights)
    # and calculated if not.
    if os.path.isfile(in_dir+'gridweights.nc'):
        grid_weights = Dataset(in_dir + 'gridweights.nc').variables['cell_weights'][:]
    else:
        grid_weights = cdo.gridweights(input=in_dir+land_file, returnArray  =  'cell_weights')
    
    # Check if population weighting is there
    pop_loc = in_dir+'{model}_pop.nc'.format(model=model)
    if os.path.isfile(pop_loc):
        pop = Dataset(pop_loc).variables['pop'][:]
        pop_weights = pop / np.sum(pop)
    else:
        return "no pop file"
    
    # Check if agriculture weighting is there
    ag_loc = in_dir+'{model}_agriculture.nc'.format(model=model)
    if os.path.isfile(ag_loc):
        ag = Dataset(ag_loc).variables['fraction'][:]
        ag_weights = ag / np.sum(ag)
    else:
        return "no ag file"
    
    """
    Prepare output dicts
    """
    
    lons_lats = {}
    masks = {}
    weights = {}
    
    """
    Output to dicts
    """
    
    # Use land file to gather lons and lats
    lons_lats['lons'] = Dataset(in_dir + land_file).variables['lon'][:]
    lons_lats['lats'] = Dataset(in_dir + land_file).variables['lat'][:]

    """
    Output masks
    """
    
    masks['global'] = np.ones_like(land)
    masks['land'] = 0.01 * land # convert 0-100 to 0-1
    masks['ocean'] = 1.0 - masks['land']
    masks['land_noice'] = 0.01 * land_noice
    
    """
    Output weights
    """
    
    # define function to calculate normalized mask weighting
    def weighted_mask(mask, weight):
        weighted_mask = mask * weight
        return weighted_mask / np.sum(weighted_mask)
    
    # Output weights all sum to 1
    weights['pop'] = pop_weights
    
    weights['ag'] = ag_weights
    
    weights['global_area'] = grid_weights
    weights['land_area'] = weighted_mask(masks['land'],grid_weights)
    weights['ocean_area'] = weighted_mask(masks['ocean'],grid_weights)
    weights['land_noice_area'] = weighted_mask(masks['land_noice'],grid_weights)
    
    # Except pop_count
    weights['pop_count'] = pop
    
    return lons_lats, masks, weights
        
def get_weight_geomip(model, mask_type=None, area=True, normalized=True):

    """
    land-sea masks, etc. are from 0 - 100%, output is 0-1
    mask_types (land file only): global, land, land_nocoast, ocean, ocean_nocoast, coast
    mask_types (with glacier file): ice, ice_alone, land_noice, land_noice_nocoase
    """

    # Define directory format
    in_dir_base = "/n/home03/pjirvine/keithfs1_pji/geomip_archive/final_data/{model}/fix/"
    #land_dir
    in_dir = in_dir_base.format(model = model)

    # Define file names for land
    nc_file_base = "{fx_type}_{model}.nc"
    land_file = nc_file_base.format(model = model, fx_type = 'sftlf')
  
    # land no ice file load
    land_noice_file = "{model}_land_no_gr_ant.nc".format(model = model)
    land_noice = Dataset(in_dir + land_noice_file).variables['sftlf'][:]

    """
    Test if land mask there and calculate grid_weights
    """

    if not os.path.isfile(in_dir+land_file):
        return "No land mask file"

    # calculate grid_weights
    # Check if gridweights file is present (this is needed as some mask files don't contain gridweights)
    # and calculated if not.
    file_loc = in_dir+ model + '_gridweights.nc'
    if os.path.isfile(file_loc):
        grid_weights = Dataset(file_loc).variables['cell_weights'][:]
        if np.isnan(grid_weights).any():
            print model_in, 'NAN present'
    else:
        print file_loc, 'gridweights not found'
        grid_weights = cdo.gridweights(input=in_dir+land_file, returnArray  =  'cell_weights')

    mask = None

    """
    Land-only masks
    """
  
    if mask_type == 'land':
        mask = Dataset(in_dir + land_file).variables['sftlf'][:]
    elif mask_type == 'land_nocoast':
        mask = 100.0 * (Dataset(in_dir + land_file).variables['sftlf'][:] > 99.9)
    elif mask_type == 'ocean':
        mask = 100.0 - Dataset(in_dir + land_file).variables['sftlf'][:]
    elif mask_type == 'ocean_nocoast':
        mask = 100.0 * (100.0 - Dataset(in_dir + land_file).variables['sftlf'][:] > 99.9)
    elif mask_type == 'coast':
        mask = 100.0 * ( (Dataset(in_dir + land_file).variables['sftlf'][:] < 99.9) * (Dataset(in_dir + land_file).variables['sftlf'][:] > 0.01) )
    elif mask_type is None or mask_type == 'global':
        mask = 100.0 * np.ones_like(Dataset(in_dir + land_file).variables['sftlf'][:])
    elif mask_type == 'land_noice':
        mask = land_noice
        
    # Output now if mask has been set or else continue
    if mask is not None:
        mask = mask * 0.01 # from 100% to fraction
        if area: # apply area weighting
            mask = mask * grid_weights
        if normalized: # normalize to 1
            mask = mask / np.sum(mask)
        if (np.amax(mask) > 1) or (np.amin(mask) < 0): # test within bounds
            return "mask error: "+str(np.amax(mask)) + ' ' + str(np.amax(mask))
        else:
            return mask

    return 'no mask name found: ' + mask_type

    # end get_weight_geomip
    
def get_all_weights(model_list, mask_list):
    
    """
    Get all weights
    """
    all_weights = {}
    for model in model_list:
        mask_data = {} # empty dict for data for all variables
        for mask in mask_list:
            mask_data[mask] = get_weight_geomip(model, mask_type=mask)
        # end var loop
        all_weights[model] = mask_data # store var dict in all_data dict.
    #end model loop

    return all_weights

"""
###
Get GeoMIP Data
###
"""

def get_2d_geomip(var, model, exp, run, seas, stat, time='11-50'):
    
    """
    Returns array of standard GeoMIP netcdf file from my archive using Dataset.
    Screens out common dimension names to return a single variable as an array.
    Will fail if there are more than one variables or a dimension has been missed.
    """
    
    # Define nc_file format
    nc_file_base = "{var}_{model}_{exp}_{run}_{time}_{seas}_{stat}.nc"
    nc_file = nc_file_base.format(var=var, model=model, exp=exp, run=run, time=time, seas=seas, stat=stat)

    # Define directory format
    in_dir_base = "/n/home03/pjirvine/keithfs1_pji/geomip_archive/final_data/{model}/{exp}/time{stat}/"
    in_dir = in_dir_base.format(model=model, exp=exp, stat=stat)
    
    file_loc = in_dir + nc_file

    # function which removes list from list.
    def take_list_from_list(longlist, list2remove):
        for item in list2remove:
            try:
                longlist.remove(item)
            except:
                pass # or say something...
        
    dim_list = ['lon', 'lon_bnds', 'lat', 'lat_bnds', 'time', 'time_bnds', u'longitude', u'latitude', 
                u'ht', u't', u't_bnds',u'surface']
    
    if os.path.isfile(file_loc):
        nc = Dataset(file_loc)
        
        vars_dims_in_nc = nc.variables.keys() # list vars and dims
        vars_in_nc = copy(vars_dims_in_nc)
        
        # remove dims from vars_dims to leave vars only
        take_list_from_list(vars_in_nc,dim_list)
        
        if len(vars_in_nc) == 1:
            var_out = vars_in_nc[0]
            return nc.variables[var_out][:]
        else:
            print "extra vars",vars_in_nc,nc_file
            return None
    
    else:
        return None # doesn't make print statement as many files are missing.

def get_data_dict(var, model, model_exp_runs, seas, nyears=40, var_offset=0.0, var_mult=1.0, fraction=0.5, ttest_level=0.05, flatten=True):

    # experiments and runs are:
    exps = ['piControl','abrupt4xCO2','G1']
    runs = [model_exp_runs[model+'_'+X][0] for X in exps] # take first run from each

    # Create data_dict

    data_dict = {}

    """
    Get means using get_2d_geomip function
    """

    piControl = get_2d_geomip(var, model, exps[0], runs[0], seas, 'mean')
    abrupt4xCO2 = get_2d_geomip(var, model, exps[1], runs[1], seas, 'mean')
    G1 = get_2d_geomip(var, model, exps[2], runs[2], seas, 'mean')

    # test if all means there
    if any ((piControl is None, abrupt4xCO2 is None, G1 is None)):
        return_list = []
        if piControl is None:
            return_list.append('_'.join([model, var, exps[0], runs[0], 'avg']))
        if abrupt4xCO2 is None:
            return_list.append('_'.join([model, var, exps[1], runs[1], 'avg']))
        if G1 is None:
            return_list.append('_'.join([model, var, exps[2], runs[2], 'avg']))
        return return_list

    # Convert zeros to 1E-29
    def zero_to_tiny(arr):
        arr[arr==0] = 1e-29
        return arr
    
    piControl = zero_to_tiny(piControl)
    abrupt4xCO2 = zero_to_tiny(abrupt4xCO2)
    G1 = zero_to_tiny(G1)
    
    # offset and multiply
    piControl = var_mult * piControl + var_offset
    abrupt4xCO2 = var_mult * abrupt4xCO2 + var_offset
    G1 = var_mult * G1 + var_offset

    """
    Get standard deviations using get_2d_geomip
    """

    piControl_std = get_2d_geomip(var, model, exps[0], runs[0], seas, 'std')
    abrupt4xCO2_std = get_2d_geomip(var, model, exps[1], runs[1], seas, 'std')
    G1_std = get_2d_geomip(var, model, exps[2], runs[2], seas, 'std')

    # test if all stds there
    if any ((piControl_std is None, abrupt4xCO2_std is None, G1_std is None)):
        return_list = []
        if piControl_std is None:
            return_list.append('_'.join([model, var, exps[0], runs[0], 'std']))
        if abrupt4xCO2_std is None:
            return_list.append('_'.join([model, var, exps[1], runs[1], 'std']))
        if G1_std is None:
            return_list.append('_'.join([model, var, exps[1], runs[2], 'std']))
        return return_list

    # Convert zeros to 1E-29
    piControl_std = zero_to_tiny(piControl_std)
    abrupt4xCO2_std = zero_to_tiny(abrupt4xCO2_std)
    G1_std = zero_to_tiny(G1_std)
    
    # multiply
    piControl_std = var_mult * piControl_std
    abrupt4xCO2_std = var_mult * abrupt4xCO2_std
    G1_std = var_mult * G1_std

    # Create G_frac - 1 = G1, 0 = abrupt4xCO2
    G_frac = abrupt4xCO2 + fraction*(G1 - abrupt4xCO2)
    G_frac_std = abrupt4xCO2_std + fraction*(G1_std - abrupt4xCO2_std)
    
    CO2_effect = abrupt4xCO2 - piControl
    srm_effect = G1 - abrupt4xCO2
    srm_CO2_effect = G1 - piControl
    frac_srm_effect = G_frac - abrupt4xCO2
    frac_srm_CO2_effect = G_frac - piControl

    """
    Produce anomalies and metrics and store in data_dict
    """

    # absolutes
    data_dict['piControl'] = piControl
    data_dict['abrupt4xCO2'] = abrupt4xCO2
    data_dict['G1'] = G1
    data_dict['G_frac'] = G_frac

    # absolute STDs
    data_dict['piControl_std'] = piControl_std
    data_dict['abrupt4xCO2_std'] = abrupt4xCO2_std
    data_dict['G1_std'] = G1_std
    data_dict['G_frac_std'] = G_frac_std

    # Calculate anomalies
    data_dict['CO2_effect'] = CO2_effect
    data_dict['srm_effect'] = srm_effect
    data_dict['srm_CO2_effect'] = srm_CO2_effect
    data_dict['frac_srm_effect'] = frac_srm_effect
    data_dict['frac_srm_CO2_effect'] = frac_srm_CO2_effect

    # Calculate % anomalies
    data_dict['CO2_effect_pc'] = 100.0 * (abrupt4xCO2 / piControl - 1.0)
    data_dict['srm_effect_pc'] = 100.0 * (G1 / abrupt4xCO2 - 1.0)
    data_dict['srm_CO2_effect_pc'] = 100.0 * (G1 / piControl - 1.0)
    data_dict['frac_srm_effect_pc'] = 100.0 * (G_frac / abrupt4xCO2 - 1.0)
    data_dict['frac_srm_CO2_effect_pc'] = 100.0 * (G_frac / piControl - 1.0)

    # Calculate SD anomalies
    data_dict['CO2_effect_SD'] = CO2_effect / piControl_std
    data_dict['srm_effect_SD'] = srm_effect / piControl_std
    data_dict['srm_CO2_effect_SD'] = srm_CO2_effect / piControl_std
    data_dict['frac_srm_effect_SD'] = frac_srm_effect / piControl_std
    data_dict['frac_srm_CO2_effect_SD'] = frac_srm_CO2_effect / piControl_std
    
    # Calculate better or worse off units
    data_dict['srm_better_off'] = abs(CO2_effect) - abs(srm_CO2_effect)
    data_dict['frac_srm_better_off'] = abs(CO2_effect) - abs(frac_srm_CO2_effect)
    data_dict['srm_better_off_SD'] = abs(data_dict['CO2_effect_SD']) - abs(data_dict['srm_CO2_effect_SD'])
    data_dict['frac_srm_better_off_SD'] = abs(data_dict['CO2_effect_SD']) - abs(data_dict['frac_srm_CO2_effect_SD'])
    data_dict['srm_better_off_pc'] = 100.0 * ( (abs(CO2_effect) - abs(srm_CO2_effect)) / piControl )
    data_dict['frac_srm_better_off_pc'] = 100.0 * ( (abs(CO2_effect) - abs(frac_srm_CO2_effect)) / piControl )
    
    # Calculate T-tests
    data_dict['CO2_ttest'] = ttest_sub(abrupt4xCO2, abrupt4xCO2_std, nyears, piControl, piControl_std, nyears) < ttest_level
    data_dict['srm_ttest'] = ttest_sub(G1, G1_std, nyears, abrupt4xCO2, abrupt4xCO2_std, nyears) < ttest_level
    data_dict['srm_CO2_ttest'] = ttest_sub(G1, G1_std, nyears, piControl, piControl_std, nyears) < ttest_level
    data_dict['frac_srm_ttest'] = ttest_sub(G_frac, G_frac_std, nyears, abrupt4xCO2, abrupt4xCO2_std, nyears) < ttest_level
    data_dict['frac_srm_CO2_ttest'] = ttest_sub(G_frac, G_frac_std, nyears, piControl, piControl_std, nyears) < ttest_level
    
    # Calculate derived T-Tests
    data_dict['srm_abs_ratio_ttest'] = ttest_sub(abs(srm_CO2_effect), G1_std, nyears,
                                                 abs(CO2_effect), abrupt4xCO2_std, nyears) < ttest_level
    data_dict['frac_srm_abs_ratio_ttest'] = ttest_sub(abs(frac_srm_CO2_effect), G1_std, nyears,
                                                      abs(CO2_effect), abrupt4xCO2_std, nyears) < ttest_level
              
    
    # Efficacy of scenarios
    data_dict['eff_G1'] = -1.0 * srm_effect / CO2_effect
    data_dict['eff_G_frac'] = -1.0 * frac_srm_effect / CO2_effect

    # Residual of scenarios from control
    data_dict['res_G1'] = srm_CO2_effect / CO2_effect
    data_dict['res_G_frac'] = frac_srm_CO2_effect / CO2_effect

    """
    Add on all classifications
    """
    
    ttest_combos = bools_x8_3ttests(data_dict['frac_srm_abs_ratio_ttest'], data_dict['CO2_ttest'], data_dict['frac_srm_CO2_ttest'])
    types, groups = types_groups_from_bools(ttest_combos, data_dict['res_G_frac'])
    
    data_dict.update(types)
    data_dict.update(groups)
    
    """
    Add on all masks and weights
    """
    
    lons_lats, masks, weights = get_masks_weights(model, variable=var)
    
    data_dict.update(lons_lats)
    data_dict.update(masks)
    data_dict.update(weights)
    
    """
    Return all metrics and variables
    """

    if flatten: # Return 1-D arrays
        # loop through all key-value pairs
        for key, value in data_dict.iteritems():
            data_dict[key] = value.flatten()
        return data_dict

    else: # return 2-D arrays
        return data_dict
    
"""
Gather all data into one large dictionary
"""

def get_all_data(var_list, model_list, model_exp_runs, seas, var_offsets=var_offsets, var_mults=var_mults, fraction=0.5,
                 ttest_level=0.05, flatten=True):
    
    """
    Loops through all models and vars in lists and gathers all summary data for each.
    Also records NONE where no data present and store list of missing input.
    """
    
    all_data = {}
    miss_list = []
    for model in model_list:

        var_data = {} # empty dict for data for all variables
        for var in var_list:

            # modify filenames of extreme indices if not NorESM
            var_filename = var
            if model != 'NorESM1-M':
                var_filename = var + var_name_mod[var]

            # Get alldata:
            temp = get_data_dict(var_filename, model, model_exp_runs, seas, var_offset=var_offsets[var],
                                 var_mult=var_mults[var], fraction=fraction, ttest_level=ttest_level, flatten=flatten)
            if type(temp) is list:
                var_data[var] = None
                miss_list.extend(temp)
            else:
                var_data[var] = temp
                    
        # end var loop
        all_data[model] = var_data # store var dict in all_data dict.
    #end model loop
    
    return all_data, miss_list

"""
Concatenation function
"""

def model_dict_cat(model_dict, metric_list=None):
    
    """
    Takes a dictionary of model : get_data_dict[metrics] and concatenates all models into one mega numpy array.
    
    Returns:
        concatenated data dictionary with values that are concatenated numpys of all models with data
        list of models combined.
    """
    
    model_list = model_dict.keys()
    
    if metric_list is None:
        # take list of metrics from first model
        metric_list = model_dict[model_list[0]].keys()
    
    model_data_list = [model_dict[model] for model in model_list]
    
    # test which models have data for this var.
    data_there = [type(X) is not list for X in model_data_list] # get_data_dict returns list if no data.
    
    # pair models with data_there result
    model_there_pairs =  zip(model_list,data_there)
    
    # make list of models with data (exclude those without)
    models_with_data = [x[0] for x in model_there_pairs if x[1]]
    
    # Make a dictionary with all metrics as keys and with concatenated model data as values
    cat_data = {}
    for metric in metric_list:
        cat_data[metric] = np.concatenate([model_dict[XXX][metric] for XXX in models_with_data])
        
    return cat_data, models_with_data

"""
Cat all data into one master dictionary
"""

def cat_data_func(all_data, model_list, var_list, metric_list):
    
    cat_data = {} # all_data but with models concatenated
    cat_models = {} # which models were included for each variable
    cat_num_models = {} # number of models included
    
    for var in var_list:

        model_data_list = [all_data[X][var] for X in model_list]

        # test which models have data for this var.
        no_data_per_var = [X is not None for X in model_data_list]

        # only include models which have data in list
        models_true_false =  zip(model_list,no_data_per_var)
        cat_models[var] = [x[0] for x in models_true_false if x[1]]

        # count number of models
        cat_num_models[var] = len(cat_models[var])

        for metric in metric_list:

            # Use cat_models dict to only concatenate models that have data together.
            cat_data[var,metric] = np.concatenate([all_data[X][var][metric] for X in cat_models[var]])
            
    return cat_data, cat_models, cat_num_models
