def ttest_sub(mean_1, std_1, nyears_1, mean_2, std_2, nyears_2, equal_var=True):

    """
    Sub-routine to call ttest_ind_from_stats from scipy
    Checks that shapes match and turns integer years into correct format
    returns pvalue.
    """

    # Convert nobs type
    nyears_1 = int(nyears_1)
    nyears_2 = int(nyears_2)

    # Create arrays like others for nobs
    nobs1_arr = (nyears_1-1) * np.ones_like(mean_1)
    nobs2_arr = (nyears_2-1) * np.ones_like(mean_1)

    """
    # ttest_ind_from_stats
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html
    """

    ttest_out = ttest_ind_from_stats(mean_1, std_1, nobs1_arr, mean_2, std_2, nobs2_arr)

    # An array of p-values matching the shape of the input arrays
    pvalue_out = ttest_out[1]

    return pvalue_out

def combination_fractions(cond_array, names, weight=None, mask_list=None):
    """
    This Function takes a list of boolean arrays with and returns the fractions where all combinations of conditions are met.
    
    weight is not None: apply a weighting when calculating the fraction.
    mask_list is not None: apply a list of masks to filter out points before calculating fraction (True = in / false = out)
        
    
    Returns:
        Dictionary {"name1_name2":[fraction], ..., "name1_..._nameN":VALUE}
    
    VALUE = [conds_only] - fraction where all conditions are met
    if mask is not None: (e.g. those areas which passed T-test)
        VALUE = [conds_only, conds_masked, conds_masked_normed, masked_fraction]
            conds_masked - fraction where all conditions met AND all masks pass
            conds_masked_normed - conds_masked normalized by fraction for which all masks pass
            masked_fraction - fraction for which all masks pass.
    """
    
    """
    Test inputs are of right shape, etc.
    """
    
    if len(cond_array) != len(names):
        return "input array lengths don't match: ", cond_array, names
    
    if weight is not None:
        if np.shape(cond_array[0]) != np.shape(weight):
            return "input array and weight not same shape: ", np.shape(cond_array[0]), np.shape(weight)
        
    if mask_list is not None:
        if len(cond_array) != len(mask_list):
            return "input array and mask_list lengths don't match: ", cond_array, mask_list
        if np.shape(cond_array[0]) != np.shape(mask_list[0]):
            return "input array and mask_list elements not same shape: ", np.shape(cond_array[0]), np.shape(mask_list[0])
    
    """
    Start processing
    """
    
    # Prepare output dictionary
    output_dict = {}
    
    # generate array of indices with which to index names and cond_array
    length = len(cond_array)
    indices = xrange(length)
    
    # Loop over N the number of inputs to combine
    for idx in xrange(length):
    
        # Generate the list of combinations of length N (idx+1)
        combinations = itertools.combinations(indices,idx+1)

        # Loop over the list of combinations of N inputs:
        for comb_list in combinations: # comb_list = list of indices for Ith combination

            """
            combine conditions and masks into one per combination
            """
            # List all combinations of conditions array
            comb_cond_list = [cond_array[X] for X in comb_list]
            
            # Combine all conditions together A AND B AND ... N
            comb_cond = reduce(lambda x,y: x*y, comb_cond_list)
            
            if mask_list is not None:
                
                # List all combinations of mask list
                comb_mask_list = [mask_list[X] for X in comb_list]
                
                # Combine together all masks A AND B AND ... N
                comb_mask = reduce(lambda x,y: x*y, comb_mask_list)
            
            """
            Calculate output
            """
            
            if weight is not None:
                # calculate weighted fraction that satisfies condition
                conds_only = np.sum(weight[comb_cond])
                
                if mask_list is not None:
                    
                    weight_masked = weight[comb_mask]
                    comb_cond_masked = comb_cond[comb_mask]
                    
                    conds_masked = np.sum(weight_masked[comb_cond_masked])
                    masked_fraction = np.sum(weight_masked)
                    if masked_fraction > 0:
                        conds_masked_normed = conds_masked / masked_fraction
                    else:
                        conds_masked_normed = 0.0   
            else:
                # calculate fraction that satisfies condition
                conds_only = 1.0*np.sum(comb_cond) / np.shape(comb_cond)[0]
                         
                if mask_list is not None:
                    
                    conds_masked = float(np.sum(comb_mask * comb_cond)) / float(np.shape(comb_cond)[0])
                    masked_fraction = float(np.sum(comb_mask)) / float(np.shape(comb_cond)[0])
                    if masked_fraction > 0:
                        conds_masked_normed = conds_masked / masked_fraction
                    else:
                        conds_masked_normed = 0.0

            """
            Produce key-value pairs for output_dict
            """
            
            # Combine names to produce key
            key = "_".join([names[X] for X in comb_list])
            
            value_dict = {}
            # EXTEND FOR MASK
            if mask_list is not None:
                value_dict['conditions'] = conds_only
                value_dict['conditions_masked'] = conds_masked
                value_dict['conditions_masked_normed'] = conds_masked_normed
                value_dict['masked'] = masked_fraction
            else:
                value_dict['conditions'] = conds_only
            
            # add element to dictionary
            output_dict[key] = value_dict
            
            # End of combination loop
        
        # End of loop over combination length
        
    return output_dict
    
    # End of combination_fractions()

"""
###
Define functions which determine better off, worse off, don't know and all sub-types
###
"""

def bools_x8_3ttests(test_1,test_2,test_3):
    
    """
    This function produces booleans for all 8 combinations of the three input T-tests:
    test_1 = abs(SRM_anom), abs(CO2_anom), SRM_std, CO2_std
    test_2 = CO2, ctrl
    test_3 = SRM, ctrl
    """
    
    return {'FFF': np.logical_not(test_1) * np.logical_not(test_2) * np.logical_not(test_3),
            'FFT': np.logical_not(test_1) * np.logical_not(test_2) * (test_3),
            'FTF': np.logical_not(test_1) * (test_2) * np.logical_not(test_3),
            'FTT': np.logical_not(test_1) * (test_2) * (test_3),
            'TFF': (test_1) * np.logical_not(test_2) * np.logical_not(test_3),
            'TFT': (test_1) * np.logical_not(test_2) * (test_3),
            'TTF': (test_1) * (test_2) * np.logical_not(test_3),
            'TTT': (test_1) * (test_2) * (test_3),}

# This snippet allows keys + values in dictionaries to be read as variables of form X.key instead of D['key']
class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def types_groups_from_bools(bool_dict, ratio):
    
    """
    This function takes the output from bools_x8_3ttests and the ratio and returns them with standard names and in groups
    
    See hypothesis document for details.
    
    ratio = SRM_anom / CO2_anom
    |>1| = exacerbated
    |<1| = moderated
    +ve = same sign
    -ve = reversed sign
    """
    
    """
        1: SRM vs CO2,   2: CO2 vs CTRL,   3: SRM vs CTRL
    """
    type_dict = {'A1': bool_dict['TFF'],
                 'A2': bool_dict['TFT'],
                 'A3': bool_dict['TTF'],
                 'A4': bool_dict['TTT'],
                 'B1': bool_dict['FFF'],
                 'B2': bool_dict['FFT'],
                 'B3': bool_dict['FTF'],
                 'B4': bool_dict['FTT'],}

    td = Bunch(type_dict)
    
    group_dict = {'all': td.A1 + td.A2 + td.A3 + td.A4 + td.B1 + td.B2 + td.B3 + td.B4,
                  'dont_know': td.A1 + td.B1 + td.B2 + td.B3 + td.B4,
                  'better_off': td.A3 + td.A4 * (abs(ratio) < 1),
                  'worse_off': td.A2 + td.A4 * (abs(ratio) > 1),
                  'not_small': td.A2 + td.A3 + td.A4 + td.B4,
                  'certain': td.A2 + td.A3 + td.A4,
                  'dont_know_small': td.A1 + td.B1 + td.B2 + td.B3,
                  'dont_know_big': td.B4,
                  'dont_know_big_over': td.B4 * (ratio < 0),
                  'better_off_perfect': td.A3,
                  'better_off_under': td.A4 * (0 < ratio) * (ratio < 1),
                  'better_off_over': td.A4 * (-1 < ratio) * (ratio < 0),
                  'worse_off_novel': td.A2,
                  'worse_off_exacerbate': td.A4 * (ratio > 1),
                  'worse_off_too_much': td.A4 * (ratio < -1),
                  'all_over': (td.A4 + td.B4) * (ratio < 0),
                 }

    return type_dict, group_dict

"""
This function calculates the fraction which are better, worse and don't know
"""

def better_worse_off(SRM_mean, SRM_std, CO2_mean, CO2_std, CTRL_mean, CTRL_std, nyears, ttest_level):
    
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
    type_dict, group_dict = types_groups_from_bools(bool_dict, ratio)
    
    return group_dict

# End def