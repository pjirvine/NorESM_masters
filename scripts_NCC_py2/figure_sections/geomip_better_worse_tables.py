
"""
This script outputs tables and to SCREEN. It does the following:
- Prepares a dictionary storing all GeoMIP better / worse off results
- Outputs TABLE with Fraction Better / Worse off - GeoMIP 50% and 100%, pop and land-noice
- Output TABLE with GeoMIP fraction better / worse off CO2 not significant
- Outputs TABLE with GEOMIP fraction with pos trends
- Outputs to SCREEN with fraction with given number of better / worse variables across GeoMIP ensemble.
"""

"""
GeoMIP dictionary to store all better / worse results
"""

def output_to_table(output_dict, out_loc):
    return pd.DataFrame.from_dict(output_dict).to_csv(out_loc)

def bw_dict_maker(model,var,exp):

    temp_dict = {}
    
    if all_data[model][var] is None:
        return temp_dict
    
    temp_bw = better_worse_off(
        all_data[model][var][exp], all_data[model][var][exp+'_std'], # SRM mean and STD
        all_data[model][var]['abrupt4xCO2'], all_data[model][var]['abrupt4xCO2_std'], # CO2 mean and STD
        all_data[model][var]['piControl'], all_data[model][var]['piControl_std'], # CTRL mean and STD
        nyears, ttest_level)
    
    temp_dict['better'] = temp_bw[0]
    temp_dict['worse'] = temp_bw[1]
    temp_dict['dont_know'] = temp_bw[2]

    temp_dict['better_nosign'] = abs(all_data[model][var][exp] - all_data[model][var]['piControl']) < abs(all_data[model][var]['abrupt4xCO2'] - all_data[model][var]['piControl'])
    temp_dict['worse_nosign'] = np.logical_not(temp_dict['better_nosign'])
    
    return temp_dict
    
#ttest settings
nyears = 40
ttest_level = 0.1

# just loop over these two
exps2list = ['G1','G_frac']

var_dict = {} # loop over lists adding to dicts
for var in var_list:
    
    exp_dict = {}
    for exp in exps2list:
    
        model_dict = {}
        for model in model_list:
            
            # return list of better, worse masks to inner dict
            model_dict[model] = bw_dict_maker(model,var,exp)
            
        exp_dict[exp] = model_dict
    var_dict[var] = exp_dict
# end loops

# [var][exp][model][mask]
bw_dict_geomip = var_dict

"""
Fraction Better / Worse off - GeoMIP 50% and 100%, pop and land-noice
"""

# This Calculates the median, mean, min and max over the GeoMIP ensemble for a given inner function and inner args
def med_min_max(inner_func, inner_args):
    
    # inner_func must read MODEL first then other args.
    # results for each model are calculated and output to an array
    model_results = [inner_func(model, *inner_args) for model in model_list]
    model_array = np.array(model_results)
    
    # The median, min and max of the model results array are returned as a list 
    return [np.nanmedian(model_array), np.nanmean(model_array), np.nanmin(model_array), np.nanmax(model_array)]

# Define function which draws out mask from better / worse dictionary and calculates weighted mean
def frac_mask(model, var, exp, mask_type, weight):
    if all_data[model][var] is None:
        return np.nan
    else:
        weighting = all_data[model][var][weight]
        mask = bw_dict_geomip[var][exp][model][mask_type]
        return 100.0 * np.sum(mask * weighting) / np.sum(weighting)
#end def
    
mask_types = ['better','worse','dont_know','better_nosign','worse_nosign']

weight_list = ['land_noice_area','pop']

exp_weight_dict = {}
for exp in exps2list:
    for weight in weight_list:
        
        var_dict = {} # loop over lists adding to dicts
        for var in var_list:
    
            mask_dict = {}
            for mask_type in mask_types:
                # Inner args are MODEL, [var, exp, mask_type, weight]
                mask_dict[mask_type] = med_min_max(frac_mask, [var,exp,mask_type,weight]) 
            var_dict[var] = mask_dict
        exp_weight_dict[exp+'_'+weight] = var_dict
        
        # Output to table within loop
        output_to_table( var_dict, table_dir+'geomip_'+exp+'_'+weight+'_bw_off.csv' )
    #end weight
#end exp

"""
GeoMIP fraction better / worse off CO2 not significant
"""

# For MASK calculate fraction of CO2 that passes
def CO2_frac_of_geomip(model, var, exp, mask_type, weight):
    
    if all_data[model][var] is None:
        return np.nan
    
    # Get CO2 ttest from all_data
    CO2_ttest = all_data[model][var]['CO2_ttest']  
    # get weighting
    weighting = all_data[model][var][weight]
    # get mask from dict
    mask = bw_dict_geomip[var][exp][model][mask_type]
    
    CO2_frac = np.sum( (CO2_ttest*mask) * weighting )
    total = np.sum( mask * weighting )
    
    return CO2_frac / total

# For CO2 pass calculate fraction of mask
def frac_of_CO2_geomip(model, var, exp, mask_type, weight):
    
    if all_data[model][var] is None:
        return np.nan
    
    # Get CO2 ttest from all_data
    CO2_ttest = all_data[model][var]['CO2_ttest']  
    # get weighting
    weighting = all_data[model][var][weight]
    # get mask from dict
    mask = bw_dict_geomip[var][exp][model][mask_type]
    
    return np.sum( mask[CO2_ttest] * weighting[CO2_ttest] ) / np.sum(weighting[CO2_ttest])

weight = 'land_noice_area'

var_dict = {}
for var in var_list:
    
    temp_dict = {}
    
    temp_dict['CO2_frac_of_better'] = med_min_max(CO2_frac_of_geomip, [var,exp,'better',weight])
    temp_dict['CO2_frac_of_worse'] = med_min_max(CO2_frac_of_geomip, [var,exp,'worse',weight])
    
    temp_dict['better_frac_of_CO2'] = med_min_max(frac_of_CO2_geomip, [var,exp,'better',weight])
    temp_dict['worse_frac_of_CO2'] = med_min_max(frac_of_CO2_geomip, [var,exp,'worse',weight])
    
    var_dict[var] = temp_dict

CO2_frac_geomip_dict = var_dict
    
output_to_table( CO2_frac_geomip_dict, table_dir+'geomip_land_noice_CO2_pass_fractions.csv') 

"""
GEOMIP fraction with pos trends
"""

# calculate masked fraction of positive trend
# model, var, exp, mask_type, weight
def masked_pos_trends(model, var, exp, mask_type, weight):
    
    if all_data[model][var] is None:
        return np.nan
        
    # trends are positive if SRM > Control
    pos_trends = all_data[model][var][exp] > all_data[model][var]['piControl']

    # get weighting
    weighting = all_data[model][var][weight]

    # get masking
    if mask_type is None:
        mask = np.ones_like(weighting, dtype=bool)
    else:
        mask = bw_dict_geomip[var][exp][model][mask_type] > 0.5
    
    # "the following models have only negative trends in gridcells made worse off:"
    # if nowhere has a positive trend, passes mask, on land:
    if np.sum(pos_trends[mask] * weighting[mask]) == 0:
        if np.sum(weighting[mask]) > 0: # Print only if some pass mask on land
            print var, exp, model, mask_type, np.sum(weighting[mask] > 0), np.sum(pos_trends[mask] * (weighting[mask] > 0))
            
    # return normalized, masked, weighted fraction of positive trend
    return 1.0 * np.sum(pos_trends[mask] * weighting[mask]) / np.sum(weighting[mask])

exp = 'G_frac'
weight = 'land_noice_area'

print """ the following GEOMIP models have only negative trends in gridcells made worse off:
"""

var_dict = {}
for var in var_list:
        
    temp_dict = {}
    
    temp_dict['all'] = med_min_max(masked_pos_trends, [var,exp,None,weight])
    temp_dict['better'] = med_min_max(masked_pos_trends, [var,exp,'better',weight])
    temp_dict['worse'] = med_min_max(masked_pos_trends, [var,exp,'worse',weight])
    
    var_dict[var] = temp_dict

geomip_pos_dict = var_dict
    
output_to_table( geomip_pos_dict, table_dir+'geomip_land_noice_pos_fractions.csv')

"""
###
###
### Begin Mega regrid script to do same better/worse combos for GeoMIP
###
###
"""
# Load modules

from netCDF4 import Dataset

# from matplotlib import cm

"""
Gather and regrid all data
"""

ref_dir = '/n/home03/pjirvine/keithfs1_pji/model_ref_files/netcdf/'

# Standard inputs for simulations
get_data_args = [model_exp_runs, 'ann']
get_data_kwargs = {'time':'11-50', 'nyears':40, 'flatten':False}

ttest = 0.1

# Regrid to this super-hi HiFLOR resolution.
out_ncfile = ref_dir + 'HiFLOR.nc'

# Dictionaries to hold each vars results for each model

abs_diff_vars = {} 
better_vars = {}

# var_list = ['tas_Amon','txxETCCDI_yr','pr_Amon','rx5dayETCCDI_yr','p-e_Amon']

for var in var_list: # loop over var_list
    
    print var
    
    # Dictionaries to hold each models results
    
    abs_diff_dict = {}
    better_dict = {}

    for model in model_list: # Loop over model_list
        
        print model

        if 'ETCCDI' in var:
            in_ncfile = ref_dir + 'NorESM1-M' + '.nc'
        else:
            in_ncfile = ref_dir + model + '.nc'

        var_filename = var
        if model != 'NorESM1-M':
            var_filename = var + var_name_mod[var]

        # Get all metrics for this model / var
        model_data = get_data_dict(var_filename, model, *get_data_args, 
                                   var_offset=var_offsets[var], var_mult=var_mults[var], fraction=fraction, 
                                   ttest_level=ttest, **get_data_kwargs)

        # Only calculated values if model_data is a dict / contains results
        if type(model_data) is dict:

            # abs_diff results:
            in_arr = abs(model_data['CO2_effect']) - abs(model_data['frac_srm_CO2_effect'])
            out_arr = cdo_regrid_array(in_ncfile, in_arr, out_ncfile)
            abs_diff_dict[model] = np.squeeze(out_arr)

            # Better / Worse Off results:
            better_worse = 1. * model_data['better_off'] -1. * model_data['worse_off']
            out_arr = cdo_regrid_array(in_ncfile, better_worse, out_ncfile)
            better_dict[model] = np.squeeze(out_arr)
    
    # End of model loop
        
    abs_diff_vars[var] = abs_diff_dict
    better_vars[var] = better_dict

# End of var loop

"""
Sum / mean over models
"""

bwoff_sum = {}
mean_abs_diff = {}

for var in var_list:

    bwoff_sum[var,'total'] = sum(better_vars[var].values())
    
    better_list = [1. * (X > 0) for X in better_vars[var].values()]
    bwoff_sum[var,'better'] = sum(better_list)
    
    worse_list = [1. * (X < 0) for X in better_vars[var].values()]
    bwoff_sum[var,'worse'] = sum(worse_list)
    
    mean_abs_diff[var] = sum(abs_diff_vars[var].values()) / len(abs_diff_dict)

    
"""
Loop over all vars and count better / worse
"""

var_list_noprecip = ['tas_Amon', 'txxETCCDI_yr', 'rx5dayETCCDI_yr', 'p-e_Amon']

# create list of models that include extremes
model_list_ext = [model for model in model_list if better_vars['txxETCCDI_yr'].get(model) is not None]

bwoff_var_sum_dict = {}
for model in model_list_ext:
    
    temp_dict = {}
        
    # create zeroed arrays to start count from
    better_count = np.zeros_like(better_vars['tas_Amon'][model])
    worse_count = np.zeros_like(better_vars['tas_Amon'][model])

    # loop over all non-precip varse
    for var in var_list_noprecip:

        # Count number of vars better / worse on each gridcell
        better_count += 1 * (better_vars[var][model] > 0)
        worse_count += 1 * (better_vars[var][model] < 0)

    temp_dict['better_count'] = better_count
    temp_dict['worse_count'] = worse_count

    bwoff_var_sum_dict[model] = temp_dict
    
"""
Define function to calculate fraction with certain number of better / worse variables across GeoMIP ensemble.
"""

def fraction_bw_geomip(b,w,weight):

    # create stores for counts
    frac_sum = 0.
    
    for model in model_list_ext:
        b_better = bwoff_var_sum_dict[model]['better_count'] == b
        w_worse = bwoff_var_sum_dict[model]['worse_count'] == w
        
        # add 1/8 (num models) of weighted average to sum total
        frac_sum += np.sum( (b_better * w_worse) * weight) / len(model_list_ext)
        
    return frac_sum
    
weight = gfdl_masks['land_noice_area']

print """
number better / worse combinations for GeoMIP, fraction of land noice area
"""
print "0W0B:", fraction_bw_geomip(0,0,weight), "0W1B:", fraction_bw_geomip(1,0,weight), "0W2B:", fraction_bw_geomip(2,0,weight), "0W3B:", fraction_bw_geomip(3,0,weight), "0W4B:", fraction_bw_geomip(4,0,weight)
print "1W0B:", fraction_bw_geomip(0,1,weight), "1W1B:", fraction_bw_geomip(1,1,weight), "1W2B:", fraction_bw_geomip(2,1,weight), "1W3B:", fraction_bw_geomip(3,1,weight)
print "2W0B:", fraction_bw_geomip(0,2,weight), "2W1B:", fraction_bw_geomip(1,2,weight), "2W2B:", fraction_bw_geomip(2,2,weight)
print "3W0B:", fraction_bw_geomip(0,3,weight), "3W1B:", fraction_bw_geomip(1,3,weight)
print "4W0B:", fraction_bw_geomip(0,4,weight)

"""
Manually copy to table!
"""