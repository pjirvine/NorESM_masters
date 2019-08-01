
"""
Basic settings
"""

nyears = 100
ttest_level = 0.1 # 90%

vars_hiflor = ['tas','tasmax','precip','precip5max','pe']

out_dir = '/n/home03/pjirvine/projects/fraction_better_off/tables/'

"""
Get weight
"""

# Use ice-free land area for weight
weight = gfdl_masks['land_noice_area'].flatten()

# Use global weight in some places
global_weight = gfdl_masks['area'].flatten()

"""
Dict to fill
"""

frac_dict = {}

frac_array = np.arange(0.,2.51,0.01)

"""
Start Loop
"""
for frac in frac_array:

    #reset inner dict
    inner_dict = {}
    
    for var in vars_hiflor:

        """
        Get raw GFDL data
        """
        
        CO2_mean = gfdl_data['CO2',var,'mean'].flatten()
        SRM_mean = gfdl_data['SRM',var,'mean'].flatten()
        CTRL_mean = gfdl_data['ctrl',var,'mean'].flatten()

        CO2_std = gfdl_data['CO2',var,'std'].flatten()
        SRM_std = gfdl_data['SRM',var,'std'].flatten()
        CTRL_std = gfdl_data['ctrl',var,'std'].flatten()
        
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
        Fill inner dict
        """
        
        inner_dict[var+'_global'] = np.sum(frac_anom * global_weight)
        inner_dict[var+'_RMS'] = RMS
        inner_dict[var+'_RMS_std'] = RMS_std
        inner_dict[var+'_mod'] = np.sum(better.flatten() * weight)
        inner_dict[var+'_exa'] = np.sum(worse.flatten() * weight)
    
    frac_dict[frac] = inner_dict
    
"""
End loop
"""

"""
Flip dictionary around
"""

flip_dict = {}

inner_keys = frac_dict[0.1].keys()

for inner_key in inner_keys:
    temp_dict = {}
    
    for outer_key, inner_dict in frac_dict.iteritems():
        temp_dict[outer_key] = inner_dict[inner_key]
    
    flip_dict[inner_key] = temp_dict

"""
Output Dict to CSV
"""

pd.DataFrame.from_dict(flip_dict).to_csv(out_dir + 'results_by_frac_geo.csv')