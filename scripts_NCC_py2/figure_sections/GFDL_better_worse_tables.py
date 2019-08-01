
"""
This script generates better / worse off tables for GFDL it does the following:

- Create a dictionary storing all better / worse results
- Output tables of Fraction Better / Worse off for GFDL 50% and 100%, pop and land-noice
- Table for Fraction of better / worse off CO2 not significant (on land noice)
- Table for GFDL fraction with pos trends [NOTE result is robustly different from GeoMIP for P-E]
- SCREEN OUTPUT - reports fraction with each better / worse combination for all variables.

"""

"""
GFDL dictionary storing all better / worse results
"""

#ttest settings
nyears = 100
ttest_level = 0.1

"""
50% Geo version
"""

bw_off_dict = {}

for var in vars_hiflor:
    
    temp_dict = {}
    
    temp_bw = better_worse_off(
            gfdl_data['SRM',var,'mean'], gfdl_data['SRM',var,'std'], # SRM mean and STD
            gfdl_data['CO2',var,'mean'], gfdl_data['CO2',var,'std'], # CO2 mean and STD
            gfdl_data['ctrl',var,'mean'], gfdl_data['ctrl',var,'std'], # CTRL mean and STD
            nyears, ttest_level)
    
    temp_dict['better'] = temp_bw[0]
    temp_dict['worse'] = temp_bw[1]
    temp_dict['dont_know'] = temp_bw[2]
    
    temp_dict['better_nosign'] = (gfdl_data['SRM',var,'mean'] - gfdl_data['CO2',var,'mean']) < (gfdl_data['CO2',var,'mean'] - gfdl_data['ctrl',var,'mean'])
    temp_dict['worse_nosign'] = np.logical_not(temp_dict['better_nosign'])
    
    bw_off_dict[var] = temp_dict

"""
100% Geo version
"""
    
bw_off_dict_100 = {}
    
for var in vars_hiflor:
    
    temp_dict = {}
    
    temp_ratio = (glbl_mean['tas']['CO2'] - glbl_mean['tas']['ctrl']) / (glbl_mean['tas']['SRM'] - glbl_mean['tas']['CO2'])
        
    SRM_100_mean = gfdl_data['CO2',var,'mean'] - temp_ratio * (gfdl_data['SRM',var,'mean'] - gfdl_data['CO2',var,'mean'])
    SRM_100_std = gfdl_data['CO2',var,'std'] - temp_ratio * (gfdl_data['SRM',var,'std'] - gfdl_data['CO2',var,'std'])
    
    temp_bw = better_worse_off(
            SRM_100_mean, SRM_100_std, # SRM mean and STD
            gfdl_data['CO2',var,'mean'], gfdl_data['CO2',var,'std'], # CO2 mean and STD
            gfdl_data['ctrl',var,'mean'], gfdl_data['ctrl',var,'std'], # CTRL mean and STD
            nyears, ttest_level)
    
    temp_dict['better'] = temp_bw[0]
    temp_dict['worse'] = temp_bw[1]
    temp_dict['dont_know'] = temp_bw[2]
    
    temp_dict['better_nosign'] = abs(SRM_100_mean - gfdl_data['ctrl',var,'mean']) < abs(gfdl_data['CO2',var,'mean'] - gfdl_data['ctrl',var,'mean'])
    temp_dict['worse_nosign'] = np.logical_not(temp_dict['better_nosign'])
    
    bw_off_dict_100[var] = temp_dict

"""
Fraction Better / Worse off - GFDL 50% and 100%, pop and land-noice
"""

def fraction_from_bw_dict(bw_dict, weight):
    
    out_dict = {} # create dictionary for output
    for key_v, values_v in bw_dict.iteritems(): # loop over outer dict (vars)
        
        inner_dict = {} # create inner dict to store results
        for key,values in bw_dict[key_v].iteritems(): # loop over inner dict (better, worse, etc.)
            
            inner_dict[key] = np.sum(bw_dict[key_v][key] * weight) # find area-weighted result
            
        out_dict[key_v] = inner_dict
        
    return out_dict

def output_to_table(output_dict, out_loc):
    return pd.DataFrame.from_dict(output_dict).to_csv(out_loc)

output_to_table( fraction_from_bw_dict(bw_off_dict, gfdl_masks['land_noice_area'] ), table_dir+'gfdl_land_noice_bw_off.csv')
output_to_table( fraction_from_bw_dict(bw_off_dict_100, gfdl_masks['land_noice_area'] ), table_dir+'gfdl_100_land_noice_bw_off.csv')
output_to_table( fraction_from_bw_dict(bw_off_dict, gfdl_masks['pop'] ), table_dir+'gfdl_pop_bw_off.csv')
output_to_table( fraction_from_bw_dict(bw_off_dict_100, gfdl_masks['pop'] ), table_dir+'gfdl_100_pop_bw_off.csv')

"""
Fraction of better / worse off CO2 not significant (on land noice)
"""

weight = gfdl_masks['land_noice_area']

CO2_frac_dict = {}

# For MASK calculate fraction of CO2 that passes
def CO2_frac_of(CO2_ttest, mask, weight):
    CO2_frac = np.sum( (CO2_ttest*mask) * weight )
    total = np.sum( mask * weight )
    return CO2_frac / total

# For CO2 pass calculate fraction of mask
def frac_of_CO2(CO2_ttest, mask, weight):
    return np.sum( mask[CO2_ttest] * weight[CO2_ttest] ) / np.sum(weight[CO2_ttest])

for var in vars_hiflor:
    
    # create masks
    CO2_pvalue = ttest_sub(gfdl_data['CO2',var,'mean'], gfdl_data['CO2',var,'std'], nyears, gfdl_data['ctrl',var,'mean'], gfdl_data['ctrl',var,'std'], nyears, equal_var=False)
    CO2_ttest = CO2_pvalue < ttest_level
    
    temp_dict = {}
    temp_dict['CO2_frac_of_better'] = CO2_frac_of(CO2_ttest, bw_off_dict[var]['better'], weight)
    temp_dict['CO2_frac_of_worse'] = CO2_frac_of(CO2_ttest, bw_off_dict[var]['worse'], weight)
    temp_dict['better_frac_of_CO2'] = frac_of_CO2(CO2_ttest, bw_off_dict[var]['better'], weight)
    temp_dict['worse_frac_of_CO2'] = frac_of_CO2(CO2_ttest, bw_off_dict[var]['worse'], weight)
    
    CO2_frac_dict[var] = temp_dict
    
output_to_table( CO2_frac_dict, table_dir+'gfdl_land_noice_CO2_pass_fractions.csv')

"""
GFDL fraction with pos trends

NOTE this has been checked and it does show 60% of worse off area sees negative trend for P-E
This is opposite to the 95% wet trend in GeoMIP.
"""

weight = gfdl_masks['land_noice_area']

pos_dict = {}

for var in vars_hiflor:
    
    # Define masks
    pos_trends = gfdl_data['SRM',var,'mean'] > gfdl_data['ctrl',var,'mean']
    neg_trends = gfdl_data['SRM',var,'mean'] < gfdl_data['ctrl',var,'mean']
    better = bw_off_dict[var]['better']
    worse = bw_off_dict[var]['worse']
    
    # Store results in temp dict
    temp_dict = {}
    temp_dict['pos_frac'] = np.sum(pos_trends * weight)
    temp_dict['better_pos_frac'] = np.sum(pos_trends[better] * weight[better]) / np.sum(weight[better])
    temp_dict['worse_pos_frac'] = np.sum(pos_trends[worse] * weight[worse]) / np.sum(weight[worse])
    temp_dict['neg_frac'] = np.sum(neg_trends * weight)
    temp_dict['better_neg_frac'] = np.sum(neg_trends[better] * weight[better]) / np.sum(weight[better])
    temp_dict['worse_neg_frac'] = np.sum(neg_trends[worse] * weight[worse]) / np.sum(weight[worse])
    pos_dict[var] = temp_dict
    
output_to_table( pos_dict, table_dir+'gfdl_land_noice_pos_fractions.csv')

"""
GFDL - Create count of better / worse
"""

count_better = np.zeros_like(1.0 * bw_off_dict['pe']['better']) 
count_worse = np.zeros_like(1.0 * bw_off_dict['pe']['better'])

vars_hiflor_noprecip = ['tas', 'tasmax', 'precip5max', 'pe']  # exclude precip

for var in vars_hiflor_noprecip: # exclude precip
    
    count_better += bw_off_dict[var]['better']
    count_worse += bw_off_dict[var]['worse']
        
def fraction_bw(b,w,weight):
    b_better = count_better == b
    w_worse = count_worse == w
    return np.sum( (b_better * w_worse) * weight)

weight = gfdl_masks['land_noice_area']

"""
Manually copy to table!
"""

print "number better / worse combinations, fraction of land noice area"
print "0W0B:", fraction_bw(0,0,weight), "0W1B:", fraction_bw(1,0,weight), "0W2B:", fraction_bw(2,0,weight), "0W3B:", fraction_bw(3,0,weight), "0W4B:", fraction_bw(4,0,weight)
print "1W0B:", fraction_bw(0,1,weight), "1W1B:", fraction_bw(1,1,weight), "1W2B:", fraction_bw(2,1,weight), "1W3B:", fraction_bw(3,1,weight)
print "2W0B:", fraction_bw(0,2,weight), "2W1B:", fraction_bw(1,2,weight), "2W2B:", fraction_bw(2,2,weight)
print "3W0B:", fraction_bw(0,3,weight), "3W1B:", fraction_bw(1,3,weight)
print "4W0B:", fraction_bw(0,4,weight)