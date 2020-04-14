# %load figure_sections/table_1.py
"""
Get better / worse off statistics for:
- land no ice area
- Full, half and perfect SG
- 90% T-Test, 80 years
"""

# output to here
table_dir = '../tables/'

weight = all_masks['land_noice_area']

case_combos = {'Full-GLENS': ['Full-GLENS','RCP8.5','Baseline'],
               'Half-GLENS': ['Half-GLENS','RCP8.5','Baseline'],
               'Perfect': ['Baseline-2','RCP8.5','Baseline'],
              }

def bwdk_format(fractions):
    better = 'better: ' + "{:4.2f}".format(fractions['better']*100.)
    worse = ' worse: ' + "{:4.2f}".format(fractions['worse']*100.)
    dont_know = ' dont_know: ' + "{:4.2f}".format(fractions['dont_know']*100.)
    return better + worse + dont_know

# create a dictionary to store output of variable loop
var_dict = {}
for var in vars_glens:
    
    # create a dict to store output of combo loop (wipes each var loop)
    combo_dict = {}
    for key, value in case_combos.items():
        sg_anom, co2_anom, sg_CO2_anom, masks, weights, fractions = better_worse_full_data(all_data, value[0], value[1], value[2], var, weight)
        combo_dict[key] = bwdk_format(fractions)
    
    # store combo dict in var dict
    var_dict[var] = combo_dict
#end fors

land_noice_out = pd.DataFrame.from_dict(var_dict).to_csv(table_dir+'GLENS_land_noice_bwoff.csv')
