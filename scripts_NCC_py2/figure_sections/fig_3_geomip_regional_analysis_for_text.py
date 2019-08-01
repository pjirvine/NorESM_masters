# %load figure_sections/geomip_regional_worse_summary.py

import ast

def ens_mean_anoms(var, region, model_list, fraction = 0.5, region_type='SREX', mask='land'):
    
    out_fileloc_base = out_dir + out_file_base.format(region=region_type, mask=mask, var=var)
    model_exp_mean_dict = pd.DataFrame.from_csv(out_fileloc_base+'_mean.csv').to_dict()

    srm_anom_list = []
    co2_anom_list = []
    
    for model in model_list:
    
        g_frac_mean = model_exp_mean_dict[model+'_abrupt4xCO2'][region] + fraction*(
                    model_exp_mean_dict[model+'_G1'][region] - model_exp_mean_dict[model+'_abrupt4xCO2'][region]) 

        srm_anom_list.append( g_frac_mean - model_exp_mean_dict[model+'_piControl'][region] )
        co2_anom_list.append( model_exp_mean_dict[model+'_abrupt4xCO2'][region] - model_exp_mean_dict[model+'_piControl'][region] )
        
    if len(srm_anom_list) == 0:
        return 'No data'
    else:
        num_pos = np.sum( np.array(srm_anom_list) > 0. )
        num_neg = np.sum( np.array(srm_anom_list) < 0. )
        
        # return mean of srm_anom_list, mean of co2_anom_list, and number of positive and negative SRM anoms
        return np.array(srm_anom_list).mean(), np.array(co2_anom_list).mean(), num_pos, num_neg
# end def    

def mean_worse_mean(var, region):
    
    data_dict = pd.DataFrame.from_csv(out_dir + 'GeoMIP_region_better_list.csv').to_dict()

    bw_list = ast.literal_eval(data_dict[var][region])
    WORSE = bw_list[3]

    WORSE_means = ens_mean_anoms(var, region, WORSE)
    all_means = ens_mean_anoms(var, region, model_list)

    print var,region
    print 'better:', len(bw_list[0]), 'worse:', bw_list[3]
    print 'all:', all_means
    print 'worse:', WORSE_means

   
"""
Print out results for all regions which are made worse off for each variable. entered manually
"""

print """
All p5max WORSE regions show a net REDUCTION in extreme precip conpared to control (GISS responsible for 4/5)
"""
mean_worse_mean('rx5dayETCCDI_yr', 'NEB')
mean_worse_mean('rx5dayETCCDI_yr', 'CEU')
mean_worse_mean('rx5dayETCCDI_yr', 'AMZ')
mean_worse_mean('rx5dayETCCDI_yr', 'EAF')

print """
WSA and SAF show many models worse for P-E. All worse models show a net INCREASE in P-E compared to control
"""
mean_worse_mean('p-e_Amon', 'WSA')
mean_worse_mean('p-e_Amon', 'SAF')

print """
other P-E not serious and all worse models show a net INCREASE in P-E compared to control
"""
mean_worse_mean('p-e_Amon', 'AMZ')
mean_worse_mean('p-e_Amon', 'WAF')

print """
AMZ and CEU serious for P. All models show a net REDUCTION in Precip compared to control.
"""
mean_worse_mean('pr_Amon', 'AMZ')
mean_worse_mean('pr_Amon', 'CEU')

print """
Other regions show one or two models worse most better for P. 
"""
mean_worse_mean('pr_Amon', 'WNA')
mean_worse_mean('pr_Amon', 'CNA')
mean_worse_mean('pr_Amon', 'ENA')
mean_worse_mean('pr_Amon', 'NEB')
mean_worse_mean('pr_Amon', 'NEU')
mean_worse_mean('pr_Amon', 'SAH')
mean_worse_mean('pr_Amon', 'WAF')
mean_worse_mean('pr_Amon', 'EAS')
mean_worse_mean('pr_Amon', 'SEA')