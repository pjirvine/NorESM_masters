
# load data and masks

var = 'pe'
var_geomip = 'p-e_Amon'

# Set percentile levels - exclude 0 and 1, this is covered by script below
pctl_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def better_pctls(CO2_anom,masks,weight,pctl_levels):
    
    # create pos and neg anom masks
    pos_CO2_mask = CO2_anom >= 0.
    neg_CO2_mask = CO2_anom < 0.

    # Calculate positive and negative percentiles
    pos_pctls = weighted_quantile(CO2_anom, pctl_levels, sample_weight=weight * pos_CO2_mask)
    neg_pctls = weighted_quantile(CO2_anom, pctl_levels, sample_weight=weight * neg_CO2_mask)

    # Combine all percentiles
    all_pctls = np.hstack((neg_pctls,0.,pos_pctls))

    # generate bounds for masks
    lower_bounds = np.hstack((-9.9e99,all_pctls))
    upper_bounds = np.hstack((all_pctls,9.9e99))

    bounds = zip(lower_bounds,upper_bounds)

    bound_masks = [np.logical_and(CO2_anom >= XXX[0], CO2_anom < XXX[1]) for XXX in bounds]

    # Fraction of area 
    def frac_area(mask, weight):
        return np.sum(weight[mask]) / np.sum(weight)

    better_fraction = [frac_area(XXX * masks['better'], XXX * weight) for XXX in bound_masks]
    worse_fraction = [frac_area(XXX * masks['worse'], XXX * weight) for XXX in bound_masks]

    return better_fraction, worse_fraction

for model in model_list:
    
    CO2_anom = all_data[model][var_geomip]['CO2_effect']
    weight = all_data[model][var_geomip]['land_noice_area']
    
    masks = {'better':all_data[model][var_geomip]['better_off'],
             'worse':all_data[model][var_geomip]['worse_off'],}
    
    better_fraction, worse_fraction = better_pctls(CO2_anom,masks,weight,pctl_levels)

    x = 0.1*np.arange(20) -1.0 + 0.1/2.0

    plt.fill_between(x, 0.*x, better_fraction,color=blue,linewidth=0.2, alpha=0.08)

weight = gfdl_masks['land_noice_area'].flatten()
CO2_anom, SRM_anom, masks, weights = hist2d_gfdl_data(gfdl_data, var, weight)
    
better_fraction, worse_fraction = better_pctls(CO2_anom,masks,weight,pctl_levels)

x = 0.1*np.arange(20) -1.0 + 0.1/2.0
plt.plot(x, better_fraction,'bo')

plt.axhline(0.,color='k')
plt.ylim(0,1)
plt.savefig(out_dir+'fig_2_geomip_better.png', format='png', dpi=480)
plt.show()
    
for model in model_list:
    
    CO2_anom = all_data[model][var_geomip]['CO2_effect']
    weight = all_data[model][var_geomip]['land_noice_area']
    
    masks = {'better':all_data[model][var_geomip]['better_off'],
             'worse':all_data[model][var_geomip]['worse_off'],}
    
    better_fraction, worse_fraction = better_pctls(CO2_anom,masks,weight,pctl_levels)

    x = 0.1*np.arange(20) -1.0 + 0.1/2.0

    plt.fill_between(x, 0.*x, worse_fraction,color=red,linewidth=0.2, alpha=0.08)
    
weight = gfdl_masks['land_noice_area'].flatten()
CO2_anom, SRM_anom, masks, weights = hist2d_gfdl_data(gfdl_data, var, weight)
    
better_fraction, worse_fraction = better_pctls(CO2_anom,masks,weight,pctl_levels)

x = 0.1*np.arange(20) -1.0 + 0.1/2.0
plt.plot(x, worse_fraction,'ro')

plt.axhline(0.,color='k')
plt.ylim(0,0.1)
plt.savefig(out_dir+'fig_2_geomip_worse.png', format='png', dpi=480)
plt.show()