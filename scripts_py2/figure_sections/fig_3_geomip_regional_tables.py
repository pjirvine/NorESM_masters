
"""
Generate tables of mean and STD for various regions for GeoMIP
"""

# Set directories
weight_dir = '/n/home03/pjirvine/keithfs1_pji/model_ref_files/weights/'
weight_file = '{model}_weights.nc'

region_dir = '/n/home03/pjirvine/projects/datasets_regions/SREX_Giorgi/geomip_masks/'
region_file = '{model}_SREX_sep.nc'

monsoon_dir = '/n/home03/pjirvine/projects/datasets_regions/monsoon_regions/monsoon_masks/'
monsoon_file = '{model}_monsoon_regions.nc'

land_dir = '/n/home03/pjirvine/keithfs1_pji/geomip_archive/final_data/{model}/fix/'
land_file = 'sftlf_{model}.nc'

data_dir = '/n/home03/pjirvine/keithfs1_pji/geomip_archive/final_data/{model}/{exp}/annual/'
data_file = '{var}{var_addon}_{model}_{exp}_{run}_{end}.nc' 
# END = 11-50_ann.nc for annual NORMAL, 11-50 for annual extremes

ends = {'tas_Amon':"11-50_ann", 'pr_Amon':"11-50_ann", 'rx5dayETCCDI_yr':"11-50", 'txxETCCDI_yr':"11-50", 'p-e_Amon':"11-50_ann"}

var_internal = {'tas_Amon':'tas','txxETCCDI_yr':'txxETCCDI','pr_Amon':'pr','rx5dayETCCDI_yr':'rx5dayETCCDI','p-e_Amon':'pr'}

SREX_abvs = ['ALA', 'CGI', 'WNA', 'CNA', 'ENA', 'CAM', 'AMZ', 'NEB', 'WSA', 'SSA', 'NEU', 'CEU', 'MED', 'SAH', 'WAF', 'EAF', 'SAF', 'NAS', 'WAS', 'CAS', 'TIB', 'EAS', 'SAS', 'SEA', 'NAU', 'SAU']
SREX_names = ['Alaska', 'Canada and Greenland', 'Western North America', 'Central North America', 'Eastern North America', 'Central America', 'Amazon', 'North Eastern Brazil', 'Western South America', 'Southern South America', 'Northern Europe', 'Central Europe', 'Mediterannean', 'Sahara', 'Western Africa', 'Eastern Africa', 'Southern Africa', 'Northern Asia', 'Western Asia', 'Central Asia', 'Tibet', 'Eastern Asia', 'Southern Asia', 'South Eastern Asia', 'Northern Australia', 'Southern Australia']

monsoon_names = ["North America", "Central America", "South America", "Sahel", "South Africa", "India", "East Asia", "North Asia", "Australasia"]
monsoon_abvs = ["NAM", "CAM", "SAM", "SAH", "SAF", "IND", "EAS", "NAS", "AUS"]

model_list = ['GISS-E2-R','HadCM3', 'BNU-ESM', 'CCSM4', 'CESM-CAM5.1-FV', 'CanESM2', 'CSIRO-Mk3L-1-2','HadGEM2-ES' ,'IPSL-CM5A-LR','MIROC-ESM','MPI-ESM-LR','NorESM1-M']
exp_list = ['piControl', 'abrupt4xCO2', 'G1']

out_dir = '/n/home03/pjirvine/projects/fraction_better_off/tables/geomip_regional_means/'
out_file_base = 'GeoMIP_{var}_{region}_{mask}' # + _mean / _std



"""
BREAK
"""


"""
Generate table of GeoMIP regional means and STDs from netcdfs
"""  

# This function gets the region masks from the file, masks them and reshapes them 
# to match the data_shape given.
def get_regions_for_mean(region_fileloc, region_name_list, data_shape, mask=None):
    
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
#END DEF

mask_list = ['global','land','ocean']

# get all masks
all_weights = get_all_weights(model_list, mask_list)

for var in var_list:

    print var
    
    model_exp_mean_dict = {}
    model_exp_std_dict = {}

    """
    model, exp Loop
    """

    for model in model_list:
        for exp in exp_list:

            # Add "_144x96" to all extreme var names except for NorESM
            if model != 'NorESM1-M':
                var_addon = var_name_mod[var]
            else:
                var_addon = ''

            run = model_exp_runs['{model}_{exp}'.format(model=model,exp=exp)][0]

            data_file_loc = (data_dir + data_file).format(model=model, exp=exp, var=var, var_addon=var_addon, run=run, end=ends[var])
            
            mean_dict=dict(zip(SREX_abvs,[0.]*len(SREX_abvs)))
            std_dict=dict(zip(SREX_abvs,[0.]*len(SREX_abvs)))
            
            # Check if data_file is present (not for extreme data for some models, use null value instead)
            if not os.path.isfile(data_file_loc):
                mean_dict = dict(zip(SREX_abvs,[-9999.]*len(SREX_abvs)))
                std_dict = dict(zip(SREX_abvs,[-9999.]*len(SREX_abvs)))
            else:
                data_nc = Dataset(data_file_loc) # make into a list to loop through? (NO loop through at higher level)
                # gather data and apply unit conversions
                # squeeze to remove degenerate dimensions.
                data_nc_data = var_mults[var] * data_nc.variables[var_internal[var]][:].squeeze() + var_offsets[var]
                
                # Test if data has correct shape
                if len(np.shape(data_nc_data)) != 3:
                    print model, exp, np.shape(data_nc_data)
                
                """
                IDEA - add option here for region / mask loop
                """

                # Switch to NorESM1-M for masks for extreme data (it's been regridded to NorESM1-M grid) 
                if 'ETCCDI' in var:
                    model_temp='NorESM1-M'
                else:
                    model_temp=model

                # Get mask
                mask = all_weights[model_temp]['land']
                
                region_fileloc = region_dir + region_file.format(model=model_temp)
                region_data_list = get_regions_for_mean(region_fileloc, SREX_abvs, np.shape(data_nc_data), mask=mask)

                # weighted (S)patial mean of regions (over time):
                region_mean_s_list = [ np.sum(data_nc_data * X, axis=(1,2)) for X in region_data_list ]

                #calculate mean and standard deviation over time.
                region_time_mean_list = [ np.mean(X) for X in region_mean_s_list ]
                region_time_std_list = [ np.std(X) for X in region_mean_s_list ]

                # Store mean and standard deviation in dict, with regions as "rows"
                mean_dict = dict(zip(SREX_abvs,region_time_mean_list))
                std_dict = dict(zip(SREX_abvs,region_time_std_list))
            #end if
            
            model_exp_mean_dict[model+'_'+exp] = mean_dict
            model_exp_std_dict[model+'_'+exp] = std_dict
                
        #end exp loop
    #end model loop

    """
    Convert dictionery to dataframe and export as CSV
    """
    
    """
    IDEA - add option here for region / mask loop
    """
    out_fileloc_base = out_dir + out_file_base.format(region='SREX', mask='land', var=var)
    pd.DataFrame.from_dict(model_exp_mean_dict).to_csv(out_fileloc_base+'_mean.csv')
    pd.DataFrame.from_dict(model_exp_std_dict).to_csv(out_fileloc_base+'_std.csv')
    
#end var loop

print """
fin
"""


"""
BREAK
"""



"""
Produce table of which GeoMIP models are: [BETTER better worse WORSE] for each region and var.
"""

frac = 0.5

#store results for each variable
var_list_dict = {}
var_num_dict = {}
var_co2_dict = {}

for var in var_list:

    print var

    out_fileloc_base = out_dir + out_file_base.format(region='SREX', mask='land', var=var)

    # load csv file and convert to dict
    model_exp_mean_dict = pd.read_csv(out_fileloc_base+'_mean.csv',index_col=0).to_dict()
    model_exp_std_dict = pd.read_csv(out_fileloc_base+'_std.csv',index_col=0).to_dict()
    
    # store / clear results for each region
    region_list_dict = {}
    region_num_dict = {}
    region_co2_dict = {}
    
    for region in SREX_abvs:

        # store / clear lists of results for models
        # define lists to hold which models are Better / worse
        B_list, b_list, w_list, W_list = [], [], [], []
        co2_sign_list, co2_insign_list = [], []
        
        for model in model_list:

            # calculate fractional G1
            g_frac_mean = model_exp_mean_dict[model+'_abrupt4xCO2'][region] + frac*(
                model_exp_mean_dict[model+'_G1'][region] - model_exp_mean_dict[model+'_abrupt4xCO2'][region])            
            
            # calculate better / worse off significance
            bw_sign = better_worse_off(
                g_frac_mean, model_exp_std_dict[model+'_G1'][region],
                model_exp_mean_dict[model+'_abrupt4xCO2'][region], model_exp_std_dict[model+'_abrupt4xCO2'][region],
                model_exp_mean_dict[model+'_piControl'][region], model_exp_std_dict[model+'_piControl'][region],
                40, 0.1)
            
            srm_abs_anom = abs(g_frac_mean - model_exp_mean_dict[model+'_piControl'][region])
            co2_abs_anom = abs(model_exp_mean_dict[model+'_abrupt4xCO2'][region] - model_exp_mean_dict[model+'_piControl'][region])
            
            # if SRM abs anom is smaller then it is better (without considering T-Test)
            bw = srm_abs_anom < co2_abs_anom
            
            # calculate CO2 significant or not
            co2_sign = ttest_sub(model_exp_mean_dict[model+'_abrupt4xCO2'][region], model_exp_std_dict[model+'_abrupt4xCO2'][region],40,
                                 model_exp_mean_dict[model+'_piControl'][region], model_exp_std_dict[model+'_piControl'][region],40) < 0.1
            
            if co2_sign:
                co2_sign_list.append(model)
            else:
                co2_insign_list.append(model)
            
            # add model to appropriate list
            if model_exp_mean_dict[model+'_piControl'][region] > -999:
                if bw_sign[2]: # if better / worse off not significant:
                    if bw: # if better:
                        b_list.append(model)
                    else: # if worse:
                        w_list.append(model)
                else: # if better / worse significant
                    if bw: # if better:
                        B_list.append(model)
                    else: # if worse:
                        W_list.append(model)
            
        # end model loop
    
        worse_n_co2 = list(set(co2_sign_list) & set(W_list)) # find overlap of CO2 sign and worse
    
        # enter results into region dict
        region_list_dict[region] = [B_list, b_list, w_list, W_list]
        region_num_dict[region] = [len(B_list),len(b_list),len(w_list),len(W_list)]
        region_co2_dict[region] = [co2_sign_list, co2_insign_list, worse_n_co2]
        
        print region, "CO2:", len(co2_sign_list), "B:", len(B_list),"b:", len(b_list),"w:", len(w_list),"W:", len(W_list), "W&CO2:", len(worse_n_co2)
    #end region loop
    
    # enter results into var dict
    var_list_dict[var] = region_list_dict
    var_num_dict[var] = region_num_dict
    var_co2_dict[var] = region_co2_dict
    
#end var loop

pd.DataFrame.from_dict(var_list_dict).to_csv(out_dir + 'GeoMIP_region_better_list.csv')
pd.DataFrame.from_dict(var_num_dict).to_csv(out_dir + 'GeoMIP_region_better_num.csv')
pd.DataFrame.from_dict(var_co2_dict).to_csv(out_dir + 'GeoMIP_co2_sign_list.csv')

print """
FIN
"""