
def bw_frac_var_seas(var, seas):
    """
    Function to return ice-free land area better / worse off for each var and season with GFDL HiFLOR
    """
    
    def get_gfdl_seas(exp,var,stat,seas):
        """
        Function to get gfdl seas data
        """
        
        jie_dir = '/n/home03/pjirvine/keithfs1_pji/gfdl_jie_data/orig_combined/'
        gfdl_file_loc = '{var}.hiFLOR.seas.mean-std.201-300.nc'  
        nc = Dataset(jie_dir + gfdl_file_loc.format(var=var))

        field_names = '{var}_{seas}_{stat}_{exp}'
        # Var = pe, precip, precip5max, tas, tasmax1max
        # seas = DJF, MAM, JJA, SON
        # stat = mean, std
        # exp = CO2, ctrl, SRM
        data = nc.variables[field_names.format(var=var, seas=seas, stat=stat, exp=exp)][:]

        return data # 2D dataset
    
    temp_bw = better_worse_off(get_gfdl_seas('SRM',var,'mean',seas), get_gfdl_seas('SRM',var,'std',seas),
                               get_gfdl_seas('CO2',var,'mean',seas), get_gfdl_seas('CO2',var,'std',seas),
                               get_gfdl_seas('ctrl',var,'mean',seas), get_gfdl_seas('ctrl',var,'std',seas),
                               100, 0.1
                              )
    
    # Apply ice-free land area weighting:
    weight = gfdl_masks['land_noice_area']
    
    # returns fraction better, fraction worse, fraction don't know
    return np.sum(temp_bw[0]*weight), np.sum(temp_bw[1]*weight), np.sum(temp_bw[2]*weight)

"""
Start main function
"""

vars_list = ['tas','tasmax1max','precip','precip5max','pe']
seas_list = ['annual','DJF','MAM','JJA','SON']

for var in vars_list:
    print '###'
    print var
    print ''
    for seas in seas_list:
        
        # Get results
        results = bw_frac_var_seas(var, seas)
        print seas, "Better:",results[0]*100.,"Worse:",results[1]*100.,"Dunno:",results[2]*100.
        