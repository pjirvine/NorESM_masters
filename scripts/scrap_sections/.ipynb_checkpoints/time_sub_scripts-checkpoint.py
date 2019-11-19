
"""
Functions needed:

closest_years_to_frac_GLENS - DONE
data for given frac

"""

def closest_years_to_frac_GLENS(frac):
    # GOAL - find the year (and associated 20-year index-range) where the GLENS - RCP8.5 anomaly is closest to the given fraction of the FULL-GLENS anomaly

    """
    Define sub-functions
    """
    
    def index_years(year, N, exp_years):
        # returns t_index for year range (N) around year
        # NOTE - only works for even N
        year_min = year - 0.5*N + 0.5
        year_max = year + 0.5*N - 0.5

        #return indices for years between min and max, inclusive
        return np.where((exp_years >= year_min) & (exp_years <= year_max))[0]

    def running_mean(x, N):
        #"efficient solution" from https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    """
    generate GLENS - RCP8.5 temperature anomaly for 20-year running mean
    """

    # get area weight
    all_masks = get_glens_masks_weights() # all_masks[masks]
    area_weight = all_masks['area'].transpose()

    # get control and feedback ensemble-mean temperature series for the 4 core ensemble members
    var = 'TREFHT'
    control_t_series = ens_timeseries(var, 'control', area_weight)
    feedback_t_series = ens_timeseries(var, 'feedback', area_weight)

    # take anomaly over period of overlap (2020), hence control starts from 10:. feedback is a couple of years longer hence :79
    anom_t_series = feedback_t_series[:79] - control_t_series[10:]

    # anom_years
    years_anom = np.array([IDX + 2020 for IDX in range(80)])

    # take running mean of anom and years
    anom_years_running_mean = running_mean(years_anom,20)
    anom_running_mean = running_mean(anom_t_series,20)

    """
    Calculate Full-GLENS anomaly for 2075-2094, and then find fraction of this value
    """

    # baseline_mean = np.mean(control_t_series[t_index_baseline]) #2010-2029 inclusive
    control_mean = np.mean(control_t_series[t_index_control]) #2075-2094 inclusive
    feedback_mean = np.mean(feedback_t_series[t_index_feedback]) #2075-2094 inclusive

    # find fractional GLENS T anomaly
    GLENS_T_anom = feedback_mean - control_mean
    frac_T_anom = GLENS_T_anom * frac
    
    """
    Find year where 20-year mean anomaly is closest to target value
    """

    # Find year where difference from frac_t_anom is smallest
    abs_diff_from_frac_T = abs(anom_running_mean - frac_T_anom)
    min_abs_diff = np.amin(abs_diff_from_frac_T)
    min_abs_diff_index = np.where(abs_diff_from_frac_T == min_abs_diff)
    min_diff_year = anom_years_running_mean[min_abs_diff_index]

    """
    return indices for 20-year period around this minimum year AND abs diff from target
    return (indices, min_abs_diff)
    """
    return index_years(min_diff_year, 20, years_anom), min_abs_diff
#end def

years_feedback[closest_years_to_frac_GLENS(0.5)[0]]
