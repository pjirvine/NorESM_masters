
"""
GFDL Load Data
"""

# import modules:

# import cf
from netCDF4 import Dataset
import numpy as np

"""
This Function gets means and standard deviations from GFDL regional data
"""

def get_means_stds(var, exp, reg_type):
    
    jie_dir = '/n/home03/pjirvine/keithfs1_pji/gfdl_jie_data/'
    gfdl_file_loc = '{var}.hiFLOR.SREX.monsoon.nc'  
    nc = Dataset(jie_dir + gfdl_file_loc.format(var=var))
    
    field_names = '{var}_{exp}_{reg_type}' #'ncvar%{var}_{exp}_{reg_type}'
    data = nc.variables[field_names.format(var=var,exp=exp,reg_type=reg_type)][:]
    
    return np.mean(data,axis=1), np.std(data,axis=1)

"""
This function gets all 2D GFDL data
"""

def get_all_gfdl():

    # Get the individual 2D datasets
    def get_gfdl(var, exp, stat):
        
        jie_file_std = '{var}_HiFLOR_{stat}_{exp}.nc'.format(var=var, exp=exp, stat=stat)
        if var == 'tasmax': # correction to account for weird naming in jie files
            var = 'tasmax1max'
        file_var = '{var}_{stat}_{exp}'.format(var=var, exp=exp, stat=stat)
        gfdl_file_loc = jie_dir + jie_file_std
        f = Dataset(gfdl_file_loc).variables[file_var][:]

        return f.squeeze()
    
    jie_dir = '/n/home03/pjirvine/keithfs1_pji/gfdl_jie_data/'

    vars_hiflor = ['tas','tasmax','precip','precip5max','pe']
    #                K      K        mm/day   mm/day    mm/day
    exps_hiflor = ['ctrl','CO2','SRM']
    stats_hiflor = ['mean','std']

    jie_data = {}

    for var in vars_hiflor:
        for exp in exps_hiflor:
            for stat in stats_hiflor:

                jie_data[exp,var,stat] = get_gfdl(var, exp, stat)

    return jie_data

"""
GFDL masks
"""

def get_gfdl_masks_weights():
    
    jie_dir = '/n/home03/pjirvine/keithfs1_pji/gfdl_jie_data/'
    
    """
    Masks
    """
    masks_weights = {}

    # turn field into array then squeeze off degenerate dimensions

    # land_noice mask
    jie_land_ga_file = 'HiFLOR_land_no_gr_ant.nc'
    f = Dataset(jie_dir + jie_land_ga_file).variables['landfrac'][:].squeeze()
    land_noice_data = f
    masks_weights['land_noice_mask'] = land_noice_data > 0.5

    # land mask
    jie_land_file = 'HiFLOR_landfrac.nc'
    f = Dataset(jie_dir + jie_land_file).variables['landfrac'][:].squeeze()
    land_data = f
    masks_weights['land_mask'] = land_data > 0.5

    """
    Weights
    """

    # pop weight
    jie_pop_file = 'HiFLOR_pop.nc'
    f = Dataset(jie_dir + jie_pop_file).variables['pop'][:].squeeze()
    pop_data = f
    masks_weights['pop'] = pop_data / np.sum(pop_data)

    # ag weight
    jie_ag_file = 'HiFLOR_agriculture.nc'
    f = Dataset(jie_dir + jie_ag_file).variables['fraction'][:].squeeze()
    ag_data = f
    masks_weights['ag'] = ag_data / np.sum(ag_data)

    # area weight
    weight_dir = '/n/home03/pjirvine/keithfs1_pji/model_ref_files/weights/'
    weight_file = 'HiFLOR_weights.nc'

    # get area weight, turn to array, squeeze off extra dims
    f = Dataset(weight_dir + weight_file).variables['cell_weights'][:].squeeze()
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
FIN
"""
