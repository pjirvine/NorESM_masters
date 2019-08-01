"""
GFDL regional analysis
"""

import pandas as pd

out_dir = '/n/home03/pjirvine/projects/fraction_better_off/tables/'

"""
Define terms for regional GFDL data
"""

variables = ['pe','precip','precip5max','tas','tasmax1max']
exps = ['ctrl','CO2','SRM']
reg_types = ['SREX','monsoon']

# Region names (ordered)

SREX_abvs = ['ALA', 'CGI', 'WNA', 'CNA', 'ENA', 'CAM', 'AMZ', 'NEB', 'WSA', 'SSA', 'NEU', 'CEU', 'MED', 'SAH', 'WAF', 'EAF', 'SAF', 'NAS', 'WAS', 'CAS', 'TIB', 'EAS', 'SAS', 'SEA', 'NAU', 'SAU']
SREX_names = ['Alaska', 'Canada and Greenland', 'Western North America', 'Central North America', 'Eastern North America', 'Central America', 'Amazon', 'North Eastern Brazil', 'Western South America', 'Southern South America', 'Northern Europe', 'Central Europe', 'Mediterannean', 'Sahara', 'Western Africa', 'Eastern Africa', 'Southern Africa', 'Northern Asia', 'Western Asia', 'Central Asia', 'Tibet', 'Eastern Asia', 'Southern Asia', 'South Eastern Asia', 'Northern Australia', 'Southern Australia']

monsoon_names = ["North America", "Central America", "South America", "Sahel", "South Africa", "India", "East Asia", "North Asia", "Australasia"]
monsoon_abvs = ["NAM", "CAM", "SAM", "SAH", "SAF", "IND", "EAS", "NAS", "AUS"]

"""
Load all regional data into dictionary
"""

# Load means and standard deviations for regions 
data = {}
for exp in exps:
    for var in variables:
        for reg_type in reg_types:
            data[exp,var,reg_type] = get_means_stds(var, exp, reg_type)
            # Data[element] = (mean(regions), std(regions))
# End fors

"""
Calculate results for each variable and region
"""

# years in simulation
nyears = 100
ttest_level = 0.1

reg_type = 'SREX'

anom = {} 
anom_pc = {} 
bw_off = {} # Elements are: (better, worse, don't know)

for var in variables:
    
    # bw_off[] : 0 = better off, 1 = worse off, 2 = dont know

    bw_temp = better_worse_off(
        data['SRM',var,reg_type][0], data['SRM',var,reg_type][1], # SRM mean and STD
        data['CO2',var,reg_type][0], data['CO2',var,reg_type][1], # CO2 mean and STD
        data['ctrl',var,reg_type][0], data['ctrl',var,reg_type][1], # CTRL mean and STD
        nyears, ttest_level)
    
    bw_temp2 = [1*x + -1*y + 0*z for x, y, z in zip(bw_temp[0],bw_temp[1],bw_temp[2])]
    bw_off[var] = dict(zip(SREX_abvs,bw_temp2))
    
    # anom[] = data[exp_1][REGION MEANS] - data[exp_2][REGION MEANS]
    CO2_anom_temp = data['CO2',var,reg_type][0] - data['ctrl',var,reg_type][0]
    SRM_anom_temp = data['SRM',var,reg_type][0] - data['ctrl',var,reg_type][0]
    anom[var,'CO2-ctrl'] = dict(zip(SREX_abvs,CO2_anom_temp))
    anom[var,'SRM-ctrl'] = dict(zip(SREX_abvs,SRM_anom_temp))
    anom[var,'better_worse'] = dict(zip(SREX_abvs,bw_temp2))
    
    # anom_pc[] = data[exp_1][REGION MEANS] -%CHANGE- data[exp_2][REGION MEANS]
    CO2_anom_pc_temp = 100.0 * ((data['CO2',var,reg_type][0] / data['ctrl',var,reg_type][0]) -1.0)
    SRM_anom_pc_temp = 100.0 * ((data['SRM',var,reg_type][0] / data['ctrl',var,reg_type][0]) -1.0)
    anom_pc[var,'CO2-ctrl'] = dict(zip(SREX_abvs,CO2_anom_pc_temp))
    anom_pc[var,'SRM-ctrl'] = dict(zip(SREX_abvs,SRM_anom_pc_temp))
    anom_pc[var,'better_worse'] = dict(zip(SREX_abvs,bw_temp2))

"""
Output to CSV via dataframe
"""

pd_bw = pd.DataFrame.from_dict(bw_off).to_csv(out_dir+'bw_off_SREX.csv')
pd_anom = pd.DataFrame.from_dict(anom).to_csv(out_dir+'anom_SREX.csv')
pd_anom_pc = pd.DataFrame.from_dict(anom_pc).to_csv(out_dir+'anom_pc_SREX.csv')