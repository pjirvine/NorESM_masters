
"""
Set mask directories and names
"""

SREX_abvs = ['ALA', 'CGI', 'WNA', 'CNA', 'ENA', 'CAM', 'AMZ', 'NEB', 'WSA', 'SSA', 'NEU', 'CEU', 'MED', 'SAH', 'WAF', 'EAF', 'SAF', 'NAS', 'WAS', 'CAS', 'TIB', 'EAS', 'SAS', 'SEA', 'NAU', 'SAU']
SREX_names = ['Alaska', 'Canada and Greenland', 'Western North America', 'Central North America', 'Eastern North America', 'Central America', 'Amazon', 'North Eastern Brazil', 'Western South America', 'Southern South America', 'Northern Europe', 'Central Europe', 'Mediterannean', 'Sahara', 'Western Africa', 'Eastern Africa', 'Southern Africa', 'Northern Asia', 'Western Asia', 'Central Asia', 'Tibet', 'Eastern Asia', 'Southern Asia', 'South Eastern Asia', 'Northern Australia', 'Southern Australia']


# This function gets the region masks from the file, masks them and reshapes them 
# to match the data_shape given.

def get_regions_for_mean(region_fileloc, region_name_list, data_shape, mask=None):
    """
    This function gets the region masks from the file, masks them and reshapes them 
    to match the data_shape given.
    """
    
    # This Sub-function normalizes the input mask
    def region_mask_norm(region_data, mask=None):
        # change from % to fraction
        region_1 = np.copy(region_data) / 100.
        # apply mask if present
        if mask is None:
            pass
        else:
            region_1 = region_1 * mask
        # normalize region
        return region_1 / np.sum(region_1)
    # End DEF
    
    # load region data
    region_nc = Dataset(region_fileloc)
    # make list of mask data for regions
    region_nc_data_list = [ region_nc.variables[ X ][:] for X in region_name_list]
    # Normalize region mask data
    region_data_n_list = [ region_mask_norm( X, mask=mask ) for X in region_nc_data_list]
    # Expand mask along time dimension to have same shape as data_nc_data
    region_data_exp_list = [ np.repeat(X, data_shape[0], axis=0) for X in region_data_n_list]
    
    return region_data_exp_list
#END DEF: get_regions_for_mean(region_fileloc, region_name_list, data_shape, mask=None)

def regional_means_stds(var, case):
    """
    This function calculates regional-mean timeseries over the SREX regions
    """
    
    data = ensemble_process(var,case, timeseries=True)[0] # [0] to select means

    # transpose data to match form of region mask data
    data = np.transpose(data)

    region_dir = '/n/home03/pjirvine/projects/datasets_regions/SREX_Giorgi/geomip_masks/'
    region_file = 'CCSM4_SREX_sep.nc'
    
    region_fileloc = region_dir + region_file
    region_data_list = get_regions_for_mean(region_fileloc, SREX_abvs, np.shape(data), mask=all_masks['land_mask'])

    # weighted (S)patial mean of regions (over time):
    region_mean_s_list = [ np.sum(data * X, axis=(1,2)) for X in region_data_list ]

    #calculate mean and standard deviation over time.
    region_time_mean_list = [ np.mean(X) for X in region_mean_s_list ]
    region_time_std_list = [ np.std(X) for X in region_mean_s_list ]

    # Store mean and standard deviation in dict, with regions as "rows"
    mean_dict = dict(zip(SREX_abvs,region_time_mean_list))
    std_dict = dict(zip(SREX_abvs,region_time_std_list))
    
    return mean_dict, std_dict
#end def: regional_means_stds(var, case)

def make_plot_data(var, regional_data_dict, anom_type='units', ttest_level=0.1, nyears=20):
    """
    ttest_level: 0.1 = 90%, nyears: 20 as ensemble mean for each year calculated, which reduces stddev
    """
    
    def num_stds_ttest(nobs, ttest_level, num=1000):
        import numpy as np
        from scipy.stats import ttest_ind_from_stats
        """
        reports number of stds to pass a t-test of a given level for a certain number of years
        ttest_level: 0.1 = 90%, 0.05 = 95%
        """

        xfactor = 1. / num

        results = np.array([ttest_ind_from_stats(X * xfactor, 1, nobs, 0, 1, nobs)[1] for X in range(num)])
        num_stds = np.array([X * xfactor for X in range(num)])

        # return number of STDs for T-Test
        return min( num_stds [ results < ttest_level ])
    
    SREX_abvs = ['ALA', 'CGI', 'WNA', 'CNA', 'ENA', 'CAM', 'AMZ', 'NEB', 'WSA', 'SSA', 'NEU', 'CEU', 'MED', 'SAH', 'WAF', 'EAF', 'SAF', 'NAS', 'WAS', 'CAS', 'TIB', 'EAS', 'SAS', 'SEA', 'NAU', 'SAU']
    SREX_names = ['Alaska', 'Canada and Greenland', 'Western North America', 'Central North America', 'Eastern North America', 'Central America', 'Amazon', 'North Eastern Brazil', 'Western South America', 'Southern South America', 'Northern Europe', 'Central Europe', 'Mediterannean', 'Sahara', 'Western Africa', 'Eastern Africa', 'Southern Africa', 'Northern Asia', 'Western Asia', 'Central Asia', 'Tibet', 'Eastern Asia', 'Southern Asia', 'South Eastern Asia', 'Northern Australia', 'Southern Australia']
    SREX_region_centres = [[-136.511,   66.277],[-57.5,  67.5],[-117.5  ,   44.283],[-95.   ,  39.283],[-72.5,  37.5],[-90.25802898,  16.60289305],[-62.05181777,  -3.75447446],[-42., -10.],[-75.89741775, -30.77603057],[-54.40601015, -38.77303345],[12.27138643, 64.46654867],[20.74534161, 50.5952795 ],[15. , 37.5],[10. , 22.5],[2.5   , 1.8175],[38.495 ,  1.8175],[ 20.995 , -23.1825],[110.,  60.],[50. , 32.5],[67.5, 40. ],[87.5, 40. ],[122.5,  35. ],[78.58108108, 17.90540541],[125.,   5.],[132.5, -20. ],[145., -40.]]
    
    # dictionary to hold plot data for each region
    SREX_plot_dict = {}
    for SREX in SREX_abvs:
        
        plot_dict = {} # temporary plot data dict
        
        plot_dict['name'] = SREX_names[SREX_abvs.index(SREX)] # use index from SREX_Abvs list to find matching entries
        plot_dict['centre'] = SREX_region_centres[SREX_abvs.index(SREX)]
        plot_dict['displace'] = [0,0] # these will be edited later
        
        # calculate various anomaly types
        
        # calculate number of STDs from control for 90% T-Test:
        num_ctrl_stds = num_stds_ttest(nyears, ttest_level)
        
        if anom_type == 'units':
            plot_dict['anom_85'] = regional_data_dict[var]['RCP8.5'][0][SREX] - regional_data_dict[var]['Baseline'][0][SREX]
            plot_dict['anom_GLENS'] = regional_data_dict[var]['Full-GLENS'][0][SREX] - regional_data_dict[var]['Baseline'][0][SREX]
            plot_dict['ttest_ctrl'] = num_ctrl_stds * regional_data_dict[var]['Baseline'][1][SREX]
        elif anom_type == 'pc':
            plot_dict['anom_85'] = 100. * ((regional_data_dict[var]['RCP8.5'][0][SREX] / regional_data_dict[var]['Baseline'][0][SREX]) - 1.0)
            plot_dict['anom_GLENS'] = 100. * ((regional_data_dict[var]['Full-GLENS'][0][SREX] / regional_data_dict[var]['Baseline'][0][SREX]) - 1.0)
            plot_dict['ttest_ctrl'] = 100. * ((num_ctrl_stds * regional_data_dict[var]['Baseline'][1][SREX]) / regional_data_dict[var]['Baseline'][0][SREX])
        elif anom_type == 'sd':
            plot_dict['anom_85'] = (regional_data_dict[var]['RCP8.5'][0][SREX] - regional_data_dict[var]['Baseline'][0][SREX]) / regional_data_dict[var]['Baseline'][1][SREX]
            plot_dict['anom_GLENS'] = (regional_data_dict[var]['Full-GLENS'][0][SREX] - regional_data_dict[var]['Baseline'][0][SREX]) / regional_data_dict[var]['Baseline'][1][SREX]
            plot_dict['ttest_ctrl'] = num_ctrl_stds
        else:
            print("anom_type not recognized: ", anom_type," please input: units, pc or sd")
            return
        
        # check whether RCP8.5 and GLENS are significantly different:
        ttest_plevel = ttest_sub(regional_data_dict[var]['RCP8.5'][0][SREX], regional_data_dict[var]['RCP8.5'][1][SREX], nyears, regional_data_dict[var]['Full-GLENS'][0][SREX], regional_data_dict[var]['Full-GLENS'][1][SREX], nyears)
        plot_dict['ttest_anoms'] = ttest_plevel < ttest_level
        
        # Evaluate type of anomaly relationship for region (e.g. better_off but flipped sign, etc.)
        plot_dict['full_type']= all_anom_relations(regional_data_dict[var]['Full-GLENS'][0][SREX], regional_data_dict[var]['Full-GLENS'][1][SREX],
                                                   regional_data_dict[var]['RCP8.5'][0][SREX], regional_data_dict[var]['RCP8.5'][1][SREX],
                                                   regional_data_dict[var]['Baseline'][0][SREX], regional_data_dict[var]['Baseline'][1][SREX],
                                                   nyears, ttest_level)
        plot_dict['half_type']= all_anom_relations(regional_data_dict[var]['Half-GLENS'][0][SREX], regional_data_dict[var]['Half-GLENS'][1][SREX],
                                                   regional_data_dict[var]['RCP8.5'][0][SREX], regional_data_dict[var]['RCP8.5'][1][SREX],
                                                   regional_data_dict[var]['Baseline'][0][SREX], regional_data_dict[var]['Baseline'][1][SREX],
                                                   nyears, ttest_level)
        
        # define function to specify number format - longwinded!
        def num_format(num, anom_type):
            from math import log10, floor
            def rounder(num, sig=3):
                return round(num, sig-int(floor(log10(abs(num))))-1)
            #enddef
            if abs(num)>=100:
                string = "{:+3.0f}".format(num)
            elif abs(num)>10:
                num_r = rounder(num, sig=3)
                string = "{:+3.1f}".format(num_r)
            elif abs(num)>1:
                num_r = rounder(num, sig=3)
                string = "{:+3.2f}".format(num_r)
            elif abs(num)>0.1:
                num_r = rounder(num, sig=2)
                string = "{:+3.2f}".format(num_r)
            else:
                num_r = rounder(num, sig=1)
                string = "{:+3.2f}".format(num_r)
            if anom_type == 'pc':
                string = string + "%"
            return string
                   
        plot_dict['anom_85_text'] = num_format(plot_dict['anom_85'],anom_type)
        plot_dict['anom_GLENS_text'] = num_format(plot_dict['anom_GLENS'],anom_type)
        
        SREX_plot_dict[SREX] = plot_dict
    # end for SREX_abvs
                   
    return SREX_plot_dict
#end def make_plot_data()

def num_region_types(plot_data):
    """
    This function returns a dictionary listing the number of regions with each type of anom relationship
    """

    # a mutually exclusive list of anomaly relationships 
    group_dict_exclusive_list=['dont_know_small','dont_know_big_none','dont_know_big_over',
                          'better_off_perfect','better_off_under','better_off_over',
                          'worse_off_novel','worse_off_exacerbate','worse_off_too_much']
    SREX_abvs = ['ALA', 'CGI', 'WNA', 'CNA', 'ENA', 'CAM', 'AMZ', 'NEB', 'WSA', 'SSA', 'NEU', 'CEU', 'MED', 'SAH', 'WAF', 'EAF', 'SAF', 'NAS', 'WAS', 'CAS', 'TIB', 'EAS', 'SAS', 'SEA', 'NAU', 'SAU']

    region_types_full = [plot_data[IDX]['full_type'] for IDX in SREX_abvs]
    region_types_half = [plot_data[IDX]['half_type'] for IDX in SREX_abvs]

    full_type_num_dict = {}
    half_type_num_dict = {}
    for anom_type in group_dict_exclusive_list:
        full_type_num_dict[anom_type] = len([IDX for IDX in region_types_full if IDX is anom_type])
        half_type_num_dict[anom_type] = len([IDX for IDX in region_types_half if IDX is anom_type])

    return full_type_num_dict, half_type_num_dict

"""
Create nested dictionary with regional means and stds.
To access data:
regional_data_dict[var][case][0/1][SREX_ABV]
[0] for mean, [1] for std
"""

"""
Create regional_data_dict
"""
case_list = ['Baseline','RCP8.5','Full-GLENS','Half-GLENS']

var_dict = {} # create dict to store loops output
for var in vars_glens:
    case_dict = {} # create dict to store loops output
    for case in case_list:
        case_dict[case] = regional_means_stds(var, case)
    var_dict[var] = case_dict

# Rename var_dict
regional_data_dict = var_dict

"""
Make data for each variable plot
"""

TREFHT_regions = make_plot_data('TREFHT', regional_data_dict, anom_type='units')
TREFHTMX_regions = make_plot_data('TREFHTMX', regional_data_dict, anom_type='units')
PRECTMX_regions = make_plot_data('PRECTMX', regional_data_dict, anom_type='units')
PRECT_regions = make_plot_data('PRECT', regional_data_dict, anom_type='units')
PE_regions = make_plot_data('P-E', regional_data_dict, anom_type='units')

"""
Specify Common plot_Region_dict updates
"""

displace_dict = {'CAM': [-5.0,-5.0],
                 'NEB': [5.0,2.0],
                 'ENA': [5.0,-5.0],
                 'WSA': [-8.,2.],
                 'WAF': [0.0,-5.0],
                 'SAF': [5.0,-10.0],
                 'SAH': [-8.0,0.0],
                 'MED': [8.0,-2.],
                 'CEU': [10.,20.0],
                 'NEU': [-8.,5.],
                 'CAS': [5.0,25.0],
                 'SAS': [0.0, -5.0],
                 'SAU': [10.,-5.],
                }

def plot_srex_region_map(var_regions,out_loc, title):
    """
    Function to plot srex region map
    """
    
    # Import
    import regionmask
    import cartopy.crs as ccrs
    
    # plot updates
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'figure.figsize': (18,9)}) # Square panels (2 across a page)
    
    def mini_panels(axis, plot_dict, half_width = 8):

        # extract values from plot_dict
        anom_1 = plot_dict['anom_85']
        anom_2 = plot_dict['anom_GLENS']
        ttest_anom = plot_dict['ttest_ctrl']
        x_loc, y_loc = plot_dict['centre']
        displace_x, displace_y = plot_dict['displace']
        text_1 = plot_dict['anom_85_text']
        text_2 = plot_dict['anom_GLENS_text']

        """
        Displace origin if needed and plot line
        """
        if plot_dict['displace'] != [0,0]:

            x_loc_orig, y_loc_orig = x_loc, y_loc

            x_loc = x_loc + displace_x
            y_loc = y_loc + displace_y

            axis.plot([x_loc,x_loc_orig],[y_loc,y_loc_orig],'k',linewidth=3, zorder=2)

        """
        Normalize anomalies for plotting
        """
        big_anom = max(abs(anom_1),abs(anom_2))
        norm_value = max(big_anom,abs(2.*ttest_anom))

        norm_anom_1 = anom_1 / norm_value
        norm_anom_2 = anom_2 / norm_value
        norm_ttest_anom = ttest_anom / norm_value

        # Set some plotting standards
        thick = 0.3
        bar_loc = 0.6
        text_shift = 0.05

        """
        Create the background and anomalies
        """
        patches = [
            # Black Border for Background
            mpatches.Rectangle((x_loc - 1.05*half_width,y_loc - 1.15*half_width), 2.1*half_width, 2.6*half_width, facecolor='k', linewidth=0, zorder=3),
            # White Background
            mpatches.Rectangle((x_loc - half_width,y_loc - 1.1*half_width), 2*half_width, 2.5*half_width, facecolor='white', linewidth=0, zorder=3),
            # Ttest grey bar
            mpatches.Rectangle((x_loc - half_width,y_loc - norm_ttest_anom * half_width), 2*half_width, 2.* norm_ttest_anom * half_width, facecolor='gray', linewidth=0, zorder=3),
            # Anom_1
            mpatches.Rectangle((x_loc - (bar_loc + 0.5*thick) * half_width,y_loc), thick*half_width, norm_anom_1 * half_width, facecolor='r', linewidth=0, zorder=4),
            # Anom_2
            mpatches.Rectangle((x_loc + (bar_loc - 0.5*thick) * half_width,y_loc), thick*half_width, norm_anom_2 * half_width, facecolor='b', linewidth=0, zorder=4),        
        ]
        for p in patches:
            axis.add_patch(p)

        """
        Add the lines
        """
        #zero line
        axis.plot([x_loc - half_width,x_loc + half_width],[y_loc,y_loc],'k',linewidth=1, zorder=5)

        #Between line
        axis.plot([x_loc - (bar_loc * half_width), x_loc + (bar_loc * half_width)],[y_loc + (norm_anom_1 * half_width),y_loc + (norm_anom_2 * half_width)],'k',linewidth=1, zorder=3)

        #Half-way Point
        axis.plot([x_loc],[y_loc + 0.5 * (norm_anom_1 + norm_anom_2) * half_width],color='purple', marker='.', markersize=12, zorder=4)

        """
        Add the text values
        """
        #text
        axis.text(x_loc - (bar_loc - text_shift) * half_width, y_loc + 1.05*half_width, text_1,  horizontalalignment='center', verticalalignment='bottom', fontsize=8, zorder=4)
        axis.text(x_loc + (bar_loc - text_shift) * half_width, y_loc + 1.05*half_width, text_2,  horizontalalignment='center', verticalalignment='bottom', fontsize=8, zorder=4)
        ### FIN ###
    #end def mini_panels()
    
    """
    Apply common updates to plot_dict
    """
    # Function to update plot_regions_dict
    def update_plot_regions(plot_regions_dict, plot_value, update_dict):
        for SREX, update_value in update_dict.items():
            plot_regions_dict[SREX][plot_value] = update_value
    #end def
    
    update_plot_regions(var_regions,'displace', displace_dict)

    """
    Create SREX mask used as base for summary plot
    """
    ax = regionmask.defined_regions.srex.plot(add_label=False, line_kws={'zorder':1, 'linewidth':1})
    plt.title(title, fontsize = 16)
    plt.tight_layout()

    """
    Plot mini-panels for each SREX region
    """
    for SREX in SREX_abvs:
        mini_panels(ax, var_regions[SREX])

    """
    Save Figure
    """    
    plt.savefig(out_loc+'.eps', format='eps', dpi=480)
    plt.savefig(out_loc+'.png', format='png', dpi=480)
    plt.show()
# end def

"""
Actually Make the Plots!
"""


out_dir = '/n/home03/pjirvine/projects/GLENS_fraction_better_off/figures/'

# Plot T
plot_srex_region_map(TREFHT_regions,out_dir + 'TREFHT_SREX_region_map', 'Surface Air Temperature (T, $^\circ$C)')

# Plot Tmax
plot_srex_region_map(TREFHTMX_regions,out_dir + 'TREFHTMX_SREX_region_map', 'Max. Surface Air Temperature (Tmax, $^\circ$C)')

# Plot P
plot_srex_region_map(PRECT_regions,out_dir + 'PRECT_SREX_region_map', 'Precipitation (P, mmDay$^{-1}$)')

# Plot Pmax
plot_srex_region_map(PRECTMX_regions,out_dir + 'PRECTMX_SREX_region_map', 'Max. Precipitation (Pmax, mmDay$^{-1}$)')

# Plot P-E
plot_srex_region_map(PE_regions,out_dir + 'P-E_SREX_region_map', 'Precipitation minus Evaporation (P-E, mmDay$^{-1}$)')
