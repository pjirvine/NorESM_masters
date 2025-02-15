"""
This python script contains analysis functions used in the fraction better off project
These functions should be of general use
"""

import numpy as np
# import cf
from scipy.stats import ttest_ind_from_stats
import itertools
from get_glens_data import *

def ens_timeseries(var, exp, weight):
    """
    Function to calculate timeseries of global means of 4-member ensemble
    """
    
    # define file_years and runs
    file_years = {'control':'201001-209912', 'feedback':'202001-209912'}
    # The same 4 core runs will be used for the RCP8.5, baseline and GLENS data
    runs = ['001','002','003','021']
    
    # Make list of 2D timeseries of data using the get_glens_annual() function.
    data_list = [get_glens_annual(var, exp, RUN, file_years[exp]) for RUN in runs]
    
    #calculate weighted mean for each member
    global_mean_list = [global_mean(DATA, weight) for DATA in data_list] # needs to be sorted !!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # match all to minimum length (some have 89 some 91 years)
    min_length = np.min([X.shape for X in global_mean_list])
    matched_length_list = [X[0:min_length] for X in global_mean_list]
    
    # ensemble mean
    global_mean_ens_array = np.vstack(matched_length_list) # shape = (4, 89)
    ens_global_mean = np.mean(global_mean_ens_array, axis=0)
    
    return ens_global_mean
#enddef ens_timeseries

def global_mean(timeseries_2D, weight):
    # Takes data of form (Lon, Lat, time) and weight of (Lon, Lat) and calculates global_mean
    
    # First match weight shape to timeseries shape
    series_shape = timeseries_2D.shape
    weight_tiled = np.tile(weight, (series_shape[2],1,1)) # repeat weight the number of years
    weight_series = np.moveaxis(weight_tiled, 0, -1)
    
    # Now multiply and sum
    weighted_timeseries = weight_series * timeseries_2D
    weighted_mean = np.sum(weighted_timeseries, (0,1))
    
    return weighted_mean
#end def global_mean

def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.

    http://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    Examples:

    weighted_quantile([1, 2, 9, 3.2, 4], [0.0, 0.5, 1.])
    array([ 1. , 3.2, 9. ])

    weighted_quantile([1, 2, 9, 3.2, 4], [0.0, 0.5, 1.], sample_weight=[2, 1, 2, 4, 1])
    array([ 1. , 3.2, 9. ])
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

    # end weighted_quantiles

def fraction_distribution(data, values, cumulative=False, sample_weight=None):
    """
    This function takes a set of data and calculates what fraction falls around the values given,
    that is [<1st, >1st and <2nd, >2nd and <3rd, ..., >Nth]

    with input:
    data = [-1,4,5,9,15]
    values = [0,10]
    the function will return:
    Return = [0.2,0.6,0.2]

    with optional sample_weight:
    sample_weight = [0.5,0.5,1.0,1.0,2.0]
    Return = [0.1,0.5,0.4]

    With cumulative option:
    Return = [0.1,0.6,1.0]
    """

    # Copy and Flatten data
    flat_data = data.copy().flatten()

    # sort weighting
    if sample_weight is not None:
        # Copy and Flatten weight
        weight = sample_weight.copy().flatten()
        # Normalize weight
        weight = weight / weight.sum()
    else:
        weight = np.ones_like(flat_data)
        weight = weight / weight.sum()

    # Calculate
    lt_1st = np.sum((flat_data < values[0]) * weight)
    gt_last = np.sum((flat_data > values[-1]) * weight)

    if len(values) == 1:
        lt_gt = [lt_1st,gt_last]
    if len(values) > 1:
        lt_gt_rest = [ np.sum((flat_data < x) * (flat_data > y) * weight) for x, y in zip(values[1:], values[:-1]) ]
        lt_gt = [lt_1st]
        lt_gt.extend(lt_gt_rest)
        lt_gt.append(gt_last)

    if cumulative:
        temp = np.cumsum(lt_gt)
        return [x if x<1.0 else 1.0 for x in temp]
    else:
        return lt_gt

    #end fraction_distribution

def sort_data_distribution(data_in, data_sort, values, distribution=False, sort_weight=None, mask=None):

    """
    Return = data_out{list of ndarrays}, fractions{list of floats}

    Sorts data_in by data_sort according to values. 2 modes of operation:

    Value-sort: data_sort is binned by values and the matching data_in grid_points are returned binned in
    the same way.
    e.g. values = [0,1] will return 3 ndarrays with values binned by data_sort into <0, 0>1, >1

    Distribution sort: data_sort is binned into percentiles (using a weighting) and this binning is applied
    to data_in.
    e.g. values = [0.25,0.5,0.25] will return 4 ndarrays with values binned by data_sort into quartiles

    NOTES:

    Values must be in ascending order

    """

    """
    Setup weighting if included:
    """
    if sort_weight is not None:
        # Copy and Flatten weight
        weight = sort_weight.copy().flatten()
        # Normalize weight
        weight = weight / weight.sum()
    else:
        weight = np.ones_like(flat_data)
        # Normalize weight
        weight = weight / weight.sum()

    """
    Distribution-sort Mode
    """
    if distribution:
        quantiles = weighted_quantile(data_sort, values, sample_weight=weight)
        values = quantiles

    print(values)

    """
    Value-Sort Mode:
    """
    #     if not distribution:

    # Record all sort values less than 1st value and greater than last
    lt_1st = data_sort < values[0]
    gt_last = data_sort > values[-1]

    if len(values) == 1:
        sort_list = [lt_1st,gt_last]
    elif len(values) > 1:
        # less than n, greater than n+1:
        lt_gt_rest = [ (data_sort < x) * (data_sort > y) for x, y in zip(values[1:], values[:-1]) ]
        # Combine all together into a list:
        sort_list = [lt_1st]
        sort_list.extend(lt_gt_rest) # extend as it's already a list
        sort_list.append(gt_last)
    else:
        return 'Values wrong length / type: ', values


    data_in_sorted = [data_in[X] for X in sort_list]
    data_frac = [np.sum(X * weight) for X in sort_list]

    return data_in_sorted, data_frac

    # end sort_data_distribution

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
                  'dont_know_big_none': td.B4 * (ratio > 0),
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
    
    return group_dict['better_off'], group_dict['worse_off'], group_dict['dont_know']
# End def

"""
This function categorizes the relationship between the 3 anomalies.
"""

def all_anom_relations(SRM_mean, SRM_std, CO2_mean, CO2_std, CTRL_mean, CTRL_std, nyears, ttest_level):
    """
    This works only for a single set of values, no numpy arrays!
    """                                                             
                                                                        
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
    
    # a mutually exclusive list of anomaly relationships 
    group_dict_exclusive_list=['dont_know_small','dont_know_big_none','dont_know_big_over',
                          'better_off_perfect','better_off_under','better_off_over',
                          'worse_off_novel','worse_off_exacerbate','worse_off_too_much']
    
    anoms_type = [IDX for IDX in group_dict_exclusive_list if group_dict[IDX]]
    if len(anoms_type) != 1:
        print('too many elements:',anoms_type)
    else:
        return anoms_type[0]
# End def


