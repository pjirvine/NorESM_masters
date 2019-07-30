
"""
This function calculates the fraction which are better, worse and don't know
"""

def full_cats(SRM_mean, SRM_std, CO2_mean, CO2_std, CTRL_mean, CTRL_std, nyears, ttest_level):
    
    # define anomalies
    CO2_anom = CO2_mean - CTRL_mean
    SRM_anom = SRM_mean - CTRL_mean

    # ratio of anomalies
    try: # check for divide by zero error and create very big number instead
        ratio = SRM_anom / CO2_anom
    except ZeroDivisionError:
        ratio = np.sign(SRM_anom) * 9.999*10**99

    # absolute_double_anom T-Test
    ttest_1 = ttest_sub(abs(SRM_anom), SRM_std, nyears,
                        abs(CO2_anom), CO2_std, nyears) < ttest_level
    # CO2, ctrl T-Test
    ttest_2 = ttest_sub(CO2_mean, CO2_std, nyears,
                        CTRL_mean, CTRL_std, nyears) < ttest_level
    # SRM, ctrl T-Test
    ttest_3 = ttest_sub(SRM_mean, SRM_std, nyears,
                        CTRL_mean, CTRL_std, nyears) < ttest_level
    
    # This geomip_data.py function returns dictionary of combinations of results
    bool_dict = bools_x8_3ttests(ttest_1,ttest_2,ttest_3)
    
    # This geomip_data.py function returns dictionary of types of results
    
    return types_groups_from_bools(bool_dict, ratio)

def print_frac(string, mask, weight, num_format='{:.2f}'):
    print string, num_format.format(100.*np.sum(mask*weight))

nyears=100
ttest_level=0.1
weight=gfdl_masks['land_noice_area']

for var in vars_hiflor:

    type_dict, group_dict = full_cats(gfdl_data['SRM',var,'mean'], gfdl_data['SRM',var,'std'], # SRM mean and STD
                                      gfdl_data['CO2',var,'mean'], gfdl_data['CO2',var,'std'], # CO2 mean and STD
                                      gfdl_data['ctrl',var,'mean'], gfdl_data['ctrl',var,'std'], # CTRL mean and STD
                                      nyears, ttest_level)

    print ""
    print ""
    print "### ",var," ###"
    print ""

    print_frac('better off frac: ',group_dict['better_off'],weight)
    print_frac('perfect: ',group_dict['better_off_perfect'],weight)
    print_frac('partial: ',group_dict['better_off_under'],weight)
    print_frac('overcompensate: ',group_dict['better_off_over'],weight)
    print ''
    print_frac('dunno frac: ',group_dict['dont_know'],weight)
    print_frac('small: ',group_dict['dont_know_small'],weight)
    print_frac('big: ',group_dict['dont_know_big'],weight)
    print_frac('big over: ',group_dict['dont_know_big_over'],weight)
    print "big under is residual!"
    print ''
    print_frac('worse off frac: ',group_dict['worse_off'],weight)
    print_frac('novel: ',group_dict['worse_off_novel'],weight)
    print_frac('exacerbate: ',group_dict['worse_off_exacerbate'],weight)
    print_frac('overcompensate: ',group_dict['worse_off_too_much'],weight)