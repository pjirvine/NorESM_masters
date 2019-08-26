"""
Code to generate input data for fraction summary figure
"""

def frac_figure_data(frac, var_list, all_data, all_masks, weight_name = 'land_noice_area', nyears=80, ttest_level=0.1):
    """
    function to generate a table of data to input to summary figure (fig "4" from NCC paper)
    """
    # Generate masks
    global_weight = all_masks['area'].flatten()
    weight = all_masks[weight_name].flatten()

    # create Dict to fill with data and then return
    inner_dict = {}
    
    #loop over all vars
    for var in var_list:

        """
        Generate Fraction-data
        """

        CO2_mean = all_data[var,'RCP8.5'][0].flatten()
        SRM_mean = all_data[var,'Full-GLENS'][0].flatten()
        CTRL_mean = all_data[var,'Baseline'][0].flatten()

        CO2_std = all_data[var,'RCP8.5'][1].flatten()
        SRM_std = all_data[var,'Full-GLENS'][1].flatten()
        CTRL_std = all_data[var,'Baseline'][1].flatten()

        """
        Generate Fraction-data
        """

        frac_mean = CO2_mean + frac*(SRM_mean - CO2_mean)
        frac_std = CO2_std + frac*(SRM_std - CO2_std)
        frac_anom = frac_mean - CTRL_mean

        """
        Generate fraction moderated / exacerbated
        """

        better, worse, dont_know = better_worse_off(frac_mean, frac_std, CO2_mean, CO2_std, CTRL_mean, CTRL_std, nyears, ttest_level)

        """
        Root-mean square
        """

        frac_anom_squared = frac_anom**2
        frac_std_anom_squared = (frac_anom / CTRL_std)**2
        RMS = ( np.sum(weight * frac_anom_squared) )**0.5
        RMS_std = ( np.sum(weight * frac_std_anom_squared) )**0.5

        """
        Fill dict with data
        """

        inner_dict[var+'_global'] = np.sum(frac_anom * global_weight)
        inner_dict[var+'_RMS'] = RMS
        inner_dict[var+'_RMS_std'] = RMS_std
        inner_dict[var+'_mod'] = np.sum(better.flatten() * weight)
        inner_dict[var+'_exa'] = np.sum(worse.flatten() * weight)
    #endfor var 
       
    # Return dict of output    
    return inner_dict
# end def frac_figure_data()

def dict_flipper(dict_to_flip):
    """
    Flips a 2-layer dictionary inside out
    """

    flip_dict = {}

    outer_keys = list(dict_to_flip.keys())
    inner_keys = list(dict_to_flip[outer_keys[0]].keys())

    for inner_key in inner_keys:
        
        temp_dict = {}

        for outer_key, inner_dict in dict_to_flip.items():
            temp_dict[outer_key] = inner_dict[inner_key]

        flip_dict[inner_key] = temp_dict
    
    # return flipped dict
    return flip_dict
#end def

"""
generate table of data for figure
"""

out_dir = "/n/home03/pjirvine/projects/GLENS_fraction_better_off/tables/"

# Create an array of output from 0 to 1.5x GLENS
frac_array = np.arange(0.,1.51,0.01)

# fill a dictionary with the output
dict_variable = {round(FRAC,2):frac_figure_data(FRAC, vars_glens, all_data, all_masks) for FRAC in frac_array}

# Flip the dictionary around to have frac as the inner element
flip_dict = dict_flipper(dict_variable)

"""
Output Dict to CSV
"""

pd.DataFrame.from_dict(flip_dict).to_csv(out_dir + 'results_by_frac_GLENS.csv')

# END
