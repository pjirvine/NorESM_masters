
def median_y_hist2d(hist2d_out, thresh_factor=50):

    """
    returns y values for median with None where less than threshold factor times the minimum weight.
    (and corresponding x values)
    """
    
    # unpack hist2d_out
    h, x, y, image = hist2d_out

    # define centres from edges
    def centres(x):
        x_1 = np.roll(x,-1)
        return ((x + x_1) / 2)[0:-1]

    x_mid = centres(x)
    y_mid = centres(y)

    # this function calculates the median
    def med_y(y_mid,x_index,weight):

        # Define threshold for calculating median and interquartiles
        min_weight = np.nanmin(h)
        med_thresh = min_weight * thresh_factor
    #     iq_thresh = min_weight * 20

        # Convert nan to zero weight
        weight = np.nan_to_num(weight)

        if np.sum(weight[x_index,:]) > med_thresh:
            return weighted_quantile(y_mid, 0.5, sample_weight=weight[x_index,:])
        else:
            return None

    median_y = [med_y(y_mid,IDX,h) for IDX in range(len(x_mid))]
    
    return x_mid, median_y

def quant_y_hist2d(hist2d_out, quant=0.5, thresh_factor=50):

    """
    returns y values for median with None where less than threshold factor times the minimum weight.
    (and corresponding x values)
    """
    
    # unpack hist2d_out
    h, x, y, image = hist2d_out

    # define centres from edges
    def centres(x):
        x_1 = np.roll(x,-1)
        return ((x + x_1) / 2)[0:-1]

    x_mid = centres(x)
    y_mid = centres(y)

    # this function calculates the median
    def quant_y(y_mid,x_index,quant,weight):

        # Define threshold for calculating median and interquartiles
        min_weight = np.nanmin(h)
        med_thresh = min_weight * thresh_factor
    #     iq_thresh = min_weight * 20

        # Convert nan to zero weight
        weight = np.nan_to_num(weight)

        if np.sum(weight[x_index,:]) > med_thresh:
            return weighted_quantile(y_mid, quant, sample_weight=weight[x_index,:])
        else:
            return None

    quant_y = [quant_y(y_mid,IDX,quant,h) for IDX in range(len(x_mid))]
    
    return x_mid, quant_y

"""
Set up plot options
"""
from matplotlib.gridspec import GridSpec

out_dir=''

out_name = 'P-E_anoms_vs_control_P-E_>10C_100%land'

title = 'All 100% land points >10C'

# apply SAT > 10 C mask
sat = True

var = 'P-E'

land_mask_100 = all_masks['land_frac'].flatten() == 100.
weight = (land_mask_100 * all_masks['land_area'].flatten()) / np.sum((land_mask_100 * all_masks['land_area'].flatten()))

thresh_factor=10.

plot_x = all_data[var,'Baseline'][0].flatten()

# # CO2 - control
# plot_y1 = ((all_data[var,'RCP8.5'][0] - all_data[var,'Baseline'][0]) / all_data[var,'Baseline'][1]).flatten()
# # SRM - control
# plot_y2 = ((all_data[var,'Full-GLENS'][0] - all_data[var,'Baseline'][0]) / all_data[var,'Baseline'][1]).flatten()
# # SRM - 4xCO2
# plot_y3 = ((all_data[var,'Full-GLENS'][0] - all_data[var,'RCP8.5'][0]) / all_data[var,'Baseline'][1]).flatten()

# CO2 - control
plot_y1 = all_data[var,'RCP8.5'][0].flatten()
# SRM - control
plot_y2 = all_data[var,'Full-GLENS'][0].flatten()
# SRM - 4xCO2
plot_y3 = all_data[var,'Full-GLENS'][0].flatten()

# apply SAT > 10C mask
if sat:
    sat_ctrl = all_data['TREFHT','Baseline'][0].flatten()
    sat_mask = sat_ctrl > 10.
    
    sat_weight = sat_mask * weight
    weight = sat_weight / np.sum(sat_weight)
    
# Dont need lowest..
bounds = [1.e-4,3.e-4,1.e-3,3.e-3,1.e-2,3.e-2,1.e-1]
labels = ['$10^{-4}$','3*$10^{-4}$','$10^{-3}$','3*$10^{-3}$','$10^{-2}$','3*$10^{-2}$','$10^{-1}$']

xlims = [-1,5]
ylims = [-1,5]

nbins = 200

cmap = plt.cm.viridis
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

"""
Set up figure
"""

f = plt.figure(figsize=cm2inch(10,16))
plt.rcParams.update({'font.size': 7})

gs = GridSpec(3, 2, width_ratios=[12,1], height_ratios=[1,1,1])
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[2,0])
cax = plt.subplot(gs[0,1])
ax4 = plt.subplot(gs[1,1])

"""
Make 2d hist plots
"""

# Plot CO2 - control anomaly
img1 = ax1.hist2d(plot_x, plot_y1, bins=nbins, range = [xlims,ylims], weights=weight, norm=norm, cmap=cmap, cmin=bounds[0], cmax=bounds[-1])

# Plot median Y value where >50x minimum weight
x,y = median_y_hist2d(img1, thresh_factor=thresh_factor)
ax1.plot(x,y,color='r',lw=0.5)

ax1.set_title(title)
ax1.set_ylabel('4xCO2 - Control P-E (mm/day)')
ax1.axhline(0.,color='k',lw=0.5)
ax1.axvline(0.,color='k',lw=0.5)

plt.axis('scaled')

# plot SRM - control anomaly
img2 = ax2.hist2d(plot_x, plot_y2, bins=nbins, range = [xlims,ylims], weights=weight, norm=norm, cmap=cmap, cmin=bounds[0], cmax=bounds[-1])

# Plot median Y value where >50x minimum weight
x,y = median_y_hist2d(img2, thresh_factor=thresh_factor)
ax2.plot(x,y,color='r',lw=0.5)

ax2.set_ylabel('Full-SG - Control P-E (mm/day)')
ax2.axhline(0.,color='k',lw=0.5)
ax2.axvline(0.,color='k',lw=0.5)

plt.axis('scaled')

# plot SRM - control anomaly
img3 = ax3.hist2d(plot_x, plot_y3, bins=nbins, range = [xlims,ylims], weights=weight, norm=norm, cmap=cmap, cmin=bounds[0], cmax=bounds[-1])

# Plot median Y value where >50x minimum weight
x,y = median_y_hist2d(img3, thresh_factor=thresh_factor)
ax3.plot(x,y,color='r',lw=0.5)

ax3.set_xlabel('Control P-E (mm/day)')
ax3.set_ylabel('Full-SG - 4xCO2 P-E (mm/day)')
ax3.axhline(0.,color='k',lw=0.5)
ax3.axvline(0.,color='k',lw=0.5)

plt.axis('scaled')

"""
Colorbar and finishing up
"""

cbar = f.colorbar(img1[3], cax=cax, ticks=bounds, format='%0.0e')
cbar.set_ticklabels(labels)
cbar.set_label('Land Area Fraction')

ax4.axis('off') # needed due to cbar plotting issue

plt.savefig(out_dir+out_name+'.png', format='png', dpi=480)
plt.savefig(out_dir+out_name+'.eps', format='eps', dpi=480)

plt.show()
