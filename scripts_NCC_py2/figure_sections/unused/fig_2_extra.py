
"""
Ratio plot
"""

def lines_5_50_95(var, weight, mask, xmin, xmax, step):
    
    def quants(values, weight):
        quantiles = [0.01,0.05,0.5,0.95,0.99]
        output = weighted_quantile(values, quantiles, sample_weight=weight)
        return output
    
    # set plot intervals
    centres = np.arange(xmin,xmax+step,step)
    lowers = np.arange(xmin-step*0.5,xmax+step*0.5,step)
    uppers = np.arange(xmin+step*0.5,xmax+step*1.5,step)
    
    #load data
    CO2_anom, SRM_anom, masks, weights = hist2d_gfdl_data(gfdl_data, var, weight)
    
    """
    Calculate absolute ratio results for all data points
    """
    
    all_ratio = SRM_anom / CO2_anom
    all_abs_ratio = abs(all_ratio)
    
    all_within_list = [(CO2_anom > lower) & (CO2_anom < upper) for lower, upper in zip(lowers,uppers)]
    all_num_list = [np.sum(X & mask) for X in all_within_list]
    all_gt_100 = np.array([X > 100 for X in all_num_list])
    all_gt_10 = np.array([X > 10 for X in all_num_list])
    
    all_quants_list = [quants(all_abs_ratio[MASK], weight[MASK]) for MASK in all_within_list ]
    
    """
    Calculate absolute ratio results for CO2 significant points
    """
    
    co2_CO2_anom = CO2_anom[masks['co2_sign']] # exclude insignificant CO2 anoms
    co2_ratio = all_ratio[masks['co2_sign']]
    co2_abs_ratio = abs(co2_ratio)
    
    co2_mask = mask[masks['co2_sign']] # mask mask and weight
    co2_weight = weight[masks['co2_sign']]
    
    co2_within_list = [(co2_CO2_anom > lower) & (co2_CO2_anom < upper) for lower, upper in zip(lowers,uppers)]
    co2_num_list = [np.sum(X & co2_mask) for X in co2_within_list]
    co2_gt_100 = np.array([X > 100 for X in co2_num_list])
    co2_gt_10 = np.array([X > 10 for X in co2_num_list])
    
    co2_quants_list = [quants(co2_abs_ratio[MASK], co2_weight[MASK]) for MASK in co2_within_list ]
    
    """
    Plot lines
    """
    
    plt.plot(centres[all_gt_10], np.array([X[2] for X in all_quants_list])[all_gt_10], color='grey')
    plt.plot(centres[all_gt_100], np.array([X[1] for X in all_quants_list])[all_gt_100], color='grey',linestyle="--")
    plt.plot(centres[all_gt_100], np.array([X[3] for X in all_quants_list])[all_gt_100], color='grey',linestyle="--")
    
    plt.plot(centres[all_gt_10], np.array([X[2] for X in co2_quants_list])[all_gt_10], color='k')
    plt.plot(centres[co2_gt_100], np.array([X[1] for X in co2_quants_list])[co2_gt_100], color='k',linestyle="--")
    plt.plot(centres[co2_gt_100], np.array([X[3] for X in co2_quants_list])[co2_gt_100], color='k',linestyle="--")
    
    plt.ylim(0,1.4)
    
    plt.axhline(1.0, color='r')
    plt.axvline(0.0, color='k')
#end def 

fig = plt.figure(figsize=cm2inch(12,12))

"""
p-e no filter
"""

axis = fig.add_subplot(221)

var = 'pe'
weight = gfdl_masks['land_noice_area'].flatten()
mask = gfdl_masks['land_noice_mask'].flatten()

plt.title('Precip - Evap (abs ratio)')
     
lines_5_50_95(var, weight, mask, -1.5, 1.5, 0.1)


axis = fig.add_subplot(222)

var = 'precip5max'

plt.title('Precip5max (abs ratio)')

lines_5_50_95(var, weight, mask, -25,25, 1)

plt.savefig(out_dir+'fig_2_extra.png', format='png', dpi=480)
plt.savefig(out_dir+'fig_2_extra.eps', format='eps', dpi=480)

plt.show()

# plt.hist(all_abs_ratio, 100, range=(-5,5), weights=weight, histtype='step')
# plt.hist(all_ratio, 100, range=(-5,5), weights=weight, histtype='step')

# axis = fig.add_subplot(212)

# co2_weight = weights['co2_sign'][masks['co2_sign']]
# plt.hist(co2_abs_ratio, 100, range=(-5,5), weights=co2_weight, histtype='step')
# plt.hist(co2_ratio, 100, range=(-5,5), weights=co2_weight, histtype='step')

# plt.show()