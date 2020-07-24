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

