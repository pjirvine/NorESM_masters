"""
This script has two parts:
- basic global means output to TABLE
- Fraction with anomaly greater than X output to SCREEN
"""

"""
GFDL Basic Global mean values
"""

table_dir = '/n/home03/pjirvine/projects/fraction_better_off/tables/'

glbl_mean = {}
land_noice_mean = {}

for var in vars_hiflor:
    exp_glbl = {}
    exp_land_noice = {}
    for exp in ['ctrl','CO2','SRM']:
        exp_glbl[exp] = np.sum(gfdl_data[exp,var,'mean'] * gfdl_masks['area'])
        exp_land_noice[exp] = np.sum(gfdl_data[exp,var,'mean'] * gfdl_masks['land_noice_area'])
        glbl_mean[var] = exp_glbl
        land_noice_mean[var] = exp_land_noice

glbl_out = pd.DataFrame.from_dict(glbl_mean).to_csv(table_dir+'gfdl_glbl_means.csv')
land_noice_out = pd.DataFrame.from_dict(land_noice_mean).to_csv(table_dir+'gfdl_land_noice_means.csv')

"""
Fraction with anomaly greater than X
"""

var = 'pe'
limit = 0.25
weight = 'land_noice_area'

def fraction_greater(var, limit,weight_name):
    
    CO2_anom = gfdl_data['CO2',var,'mean'] - gfdl_data['ctrl',var,'mean']
    SRM_anom = gfdl_data['SRM',var,'mean'] - gfdl_data['ctrl',var,'mean']
    weight = gfdl_masks[weight_name]
    
    print "CO2 anom > ", limit," = ", 100. * np.sum((CO2_anom > limit) * weight), "%"
    print "CO2 anom < ", -1.*limit," = ", 100. * np.sum((CO2_anom < -1*limit) * weight), "%"
    print "SRM anom > ", limit," = ", 100. * np.sum((SRM_anom > limit) * weight), "%"
    print "SRM anom < ", -1.*limit," = ", 100. * np.sum((SRM_anom < -1*limit) * weight), "%"
    
fraction_greater(var, limit, weight)