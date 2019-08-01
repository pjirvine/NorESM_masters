
"""
GFDL 2D histogram
"""

from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import gridspec
import matplotlib.patches as patches

"""
Define function to get data in format for plot
"""

def hist2d_gfdl_data(gfdl_data, var, weight, frac_100=False):

    nyears = 100
    ttest_level = 0.1 # 90%
    
    CO2_mean = gfdl_data['CO2',var,'mean']
    SRM_mean = gfdl_data['SRM',var,'mean']
    CTRL_mean = gfdl_data['ctrl',var,'mean']

    CO2_std = gfdl_data['CO2',var,'std']
    SRM_std = gfdl_data['SRM',var,'std']
    CTRL_std = gfdl_data['ctrl',var,'std']

    CO2_anom = CO2_mean - CTRL_mean
    SRM_anom = SRM_mean - CTRL_mean

    # If frac_100 modify the SRM results
    if frac_100:
        # Double difference between 50%Geo and 2xCO2 (from 0.93 to 1.86 of 2 C warming offset)
        SRM_mean_100 = CO2_mean + 2.0*(SRM_mean - CO2_mean)
        SRM_std_100 = CO2_std + 2.0*(SRM_std - CO2_std)
        
        SRM_mean = SRM_mean_100
        SRM_std = SRM_std_100
        SRM_anom = SRM_mean - CTRL_mean
    
    b_nosign = abs(SRM_anom) < abs(CO2_anom)
    w_nosign = abs(SRM_anom) >= abs(CO2_anom)
    
    co2_sign = ttest_sub(CO2_mean, CO2_std, nyears,
                        CTRL_mean, CTRL_std, nyears) < ttest_level
    
    # Returns better[], worse[], dont_know[]
    better, worse, dont_know = better_worse_off(SRM_mean, SRM_std, CO2_mean, CO2_std, CTRL_mean, CTRL_std, nyears, ttest_level)
        
    certain = better + worse
    
    masks = {}
    weights = {}
    
    masks['better'] = better.flatten()
    masks['worse'] = worse.flatten()
    masks['dont_know'] = dont_know.flatten()
    masks['certain'] = certain.flatten()
    masks['b_nosign'] = b_nosign.flatten()
    masks['w_nosign'] = w_nosign.flatten()
    masks['co2_sign'] = co2_sign.flatten()
    
    def weight_func(mask, weight):
        return weight.flatten() * mask.flatten()
    
    weights['better'] = weight_func(better,weight)
    weights['worse'] = weight_func(worse,weight)
    weights['dont_know'] = weight_func(dont_know,weight)
    weights['certain'] = weight_func(certain,weight)
    weights['b_nosign'] = weight_func(b_nosign,weight)
    weights['w_nosign'] = weight_func(w_nosign,weight)
    weights['co2_sign'] = weight_func(co2_sign,weight)
    
#     certain_weight = weight.flatten() * certain.flatten()
#     certain_weight = certain_weight / np.sum(certain_weight)
    
    return CO2_anom.flatten(), SRM_anom.flatten(), masks, weights

def sort_axes(axis, xlims, ylims, xnum_steps, ynum_steps, num_format='%0.1f'):
    
    axis.set_xlim(xlims)
    axis.set_ylim(ylims)

    axis.set_yticks(np.linspace(ylims[0],ylims[1],xnum_steps))
    axis.set_xticks(np.linspace(ylims[0],ylims[1],ynum_steps))

    axis.xaxis.set_major_formatter(ticker.FormatStrFormatter(num_format))
    axis.yaxis.set_major_formatter(ticker.FormatStrFormatter(num_format))

def add_lines(axis):
    axis.axhline(0, color='k',zorder=0, lw=0.6)
    axis.axvline(0, color='k',zorder=0, lw=0.6)
    axis.plot(xlims,xlims, color='k',zorder=0, lw=0.6)
    axis.plot([xlims[0],-1.*xlims[0]],[-1.*xlims[0],xlims[0]], color='k',zorder=0, lw=0.6)
    
def bw_off_plot_func(CO2_anom,values,mask,weight):
    
    # Calculate fraction of CO2 anom distribution across values with weighted mask
    bwd = np.array(fraction_distribution(CO2_anom[mask], values, sample_weight=weight[mask]))
    # Return 100* fraction of points at each interval that are masked
    return 100 * bwd / (total * np.sum(weight) / np.sum(weight[mask]))
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    #https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    
"""
settings
"""

out_dir = '/n/home03/pjirvine/projects/fraction_better_off/figures/fig_2/'

frac_100 = False
background = True
line_1_99_top = True
line_1_99_bottom = True
angled_labels = False

"""
Get mask
"""

weight = gfdl_masks['land_noice_area'].flatten()

"""
Common settings
"""

bounds = [1.e-6,1.e-5,1.e-4,1.e-3,1.e-2,1.e-1]
labels = ['$10^{-6}$','$10^{-5}$','$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$']

cmap = plt.cm.viridis
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# g_cmap = plt.cm.gist_gray
# g_norm = mpl.colors.BoundaryNorm(bounds, g_cmap.N)

nbins = 200

"""
Create figure
"""

fig = plt.figure(figsize=cm2inch(15,13))
plt.rcParams.update({'font.size': 8})

"""
Start TOP
"""

"""
p-e no filter
"""

axis = fig.add_subplot(221)
ax_pe_top = axis

var = 'pe'

# plt.title('Precipitation - Evaporation (PE)')

plt.axis('scaled')

CO2_anom, SRM_anom, masks, weights = hist2d_gfdl_data(gfdl_data, var, weight, frac_100=frac_100)

# Set axes and add lines

xlims = [-1.5,1.5]
ylims = xlims
xnum_steps, ynum_steps = 7, 7

sort_axes(axis, xlims, ylims, xnum_steps, ynum_steps, num_format='%0.1f')

add_lines(axis)

# produce plot

img = axis.hist2d(CO2_anom, SRM_anom, bins=nbins, range = [xlims,ylims], weights=weight, norm=norm, cmap=cmap, cmin=bounds[0], cmax=bounds[-1])
# img = ax_pe.hist2d(CO2_anom[certain], SRM_anom[certain], bins=100, range = [xlims,ylims], weights=weight[certain], norm=norm, cmap=cmap, cmin=1.e-12)

if line_1_99_top:
    range_1_99 = weighted_quantile(CO2_anom, [0.01,0.99], sample_weight=weight)
    
    plt.plot([range_1_99[0],range_1_99[0]],[1.5,-2.5],color='0.5', lw=0.6, clip_on=False)
    plt.plot([range_1_99[1],range_1_99[1]],[1.5,-2.5],color='0.5', lw=0.6, clip_on=False)
    
    plt.plot([range_1_99[0],range_1_99[1]],[-1.8,-1.8],color='0.5', lw=0.6, clip_on=False)
#     plt.text( (range_1_99[0]+range_1_99[1])/2.0,-1.8,'98%',va='center',ha='center',
#              clip_on=False, color='black', bbox=dict(facecolor='white', edgecolor='white'))

if angled_labels:
    top_left = [-0.8,0.8]
    bottom_right = [0.8,-0.8]
    shift = 0.2
    plt.text(top_left[0]-shift,top_left[1],'moderated',ha='center',va='center',rotation=-45)
    plt.text(top_left[0],top_left[1]+shift,'exacerbated',ha='center',va='center',rotation=-45)
    plt.text(bottom_right[0]+shift,bottom_right[1],'moderated',ha='center',va='center',rotation=-45)
    plt.text(bottom_right[0],bottom_right[1]-shift,'exacerbated',ha='center',va='center',rotation=-45)
    
# if frac_100:
#     plt.ylabel('Full-SG anomaly (mmd$^{-1}$)')
# else:
#     plt.ylabel('Half-SG anomaly (mmd$^{-1}$)')

if background:
    axis.set_facecolor(l_red)
    axis.fill_between(xlims, ylims, [-1.*ylims[0],-1.*ylims[1]], color=l_blue, zorder=0, lw=0)
    
"""
precip 5 max
"""

axis = fig.add_subplot(222)
ax_p5_top = axis

var = 'precip5max'

# plt.title('Max. 5-day Precipitation (Px)')

plt.axis('scaled')



CO2_anom, SRM_anom, masks, weights = hist2d_gfdl_data(gfdl_data, var, weight, frac_100=frac_100)

# Set axes and add lines

xlims = [-25,25]
ylims = xlims
xnum_steps, ynum_steps = 5, 5

sort_axes(axis, xlims, ylims, xnum_steps, ynum_steps, num_format='%0.0f')

add_lines(axis)

# produce plot

img = axis.hist2d(CO2_anom, SRM_anom, bins=nbins, range = [xlims,ylims], weights=weight, norm=norm, cmap=cmap, cmin=bounds[0], cmax=bounds[-1])

if line_1_99_top:
    range_1_99 = weighted_quantile(CO2_anom, [0.01,0.99], sample_weight=weight)

    plt.plot([range_1_99[0],range_1_99[0]],[25,-35],color='0.5', lw=0.6, clip_on=False)
    plt.plot([range_1_99[1],range_1_99[1]],[25,-35],color='0.5', lw=0.6, clip_on=False)
    
    plt.plot([range_1_99[0],range_1_99[1]],[-30,-30],color='0.5', lw=0.6, clip_on=False)
#     plt.text( (range_1_99[0]+range_1_99[1])/2.0,-30,'98%',va='center',ha='center',
#              clip_on=False, color='black', bbox=dict(facecolor='white', edgecolor='white'))

if angled_labels:
    px_tp_pe = 25./1.5 # reproduce same proportions as for PE using this factor
    top_left = [-0.8*px_tp_pe,0.8*px_tp_pe]
    bottom_right = [0.8*px_tp_pe,-0.8*px_tp_pe]
    shift = 0.2*px_tp_pe
    plt.text(top_left[0]-shift,top_left[1],'moderated',ha='center',va='center',rotation=-45)
    plt.text(top_left[0],top_left[1]+shift,'exacerbated',ha='center',va='center',rotation=-45)
    plt.text(bottom_right[0]+shift,bottom_right[1],'moderated',ha='center',va='center',rotation=-45)
    plt.text(bottom_right[0],bottom_right[1]-shift,'exacerbated',ha='center',va='center',rotation=-45)
    
if background:
    axis.set_facecolor(l_red)
    axis.fill_between(xlims, ylims, [-1.*ylims[0],-1.*ylims[1]], color=l_blue, zorder=0, lw=0)
    
"""
Create BOTTOM figure
"""

"""
p-e no filter
"""

axis = fig.add_subplot(223)
ax_pe_mid = axis

# plt.title('Precip -Evap (mmd$^{-1}$)')

var = 'pe'

CO2_anom, SRM_anom, masks, weights = hist2d_gfdl_data(gfdl_data, var, weight, frac_100=frac_100)

xmin = -1.5
xmax = 1.5
step = 0.05

values = np.arange(xmin, xmax+step, step) # need to extend just beyond end to add endpoint
centres = np.arange(xmin-step/2, xmax+step, step)

plt.ylim(0,100)
plt.xlim(xmin,xmax)

total = np.array(fraction_distribution(CO2_anom, values, cumulative=False, sample_weight=weight))

better_plot = bw_off_plot_func(CO2_anom,values,masks['better'],weight)
b_nosign_plot = bw_off_plot_func(CO2_anom,values,masks['b_nosign'],weight) # better including non significant results
w_nosign_plot = bw_off_plot_func(CO2_anom,values,masks['w_nosign'],weight) # worse "" ""
worse_plot = bw_off_plot_func(CO2_anom,values,masks['worse'],weight)

# plot better
plt.fill_between(centres,100,100-b_nosign_plot,color=l_blue,lw=0)
plt.fill_between(centres,100,100-better_plot,color=blue,lw=0)
# plot worse
plt.fill_between(centres,0,w_nosign_plot,color=l_red,lw=0) # lw = 0 removes line at edges
plt.plot(centres,w_nosign_plot,color='k',lw=0.6)
plt.fill_between(centres,0,worse_plot,color=red,lw=0) # lw = 0 removes line at edges

if line_1_99_bottom:
    range_1_99 = weighted_quantile(CO2_anom, [0.01,0.99], sample_weight=weight)
    plt.axvline(x=range_1_99[0],color='0.5', lw=0.6)
    plt.axvline(x=range_1_99[1],color='0.5', lw=0.6)

# plt.ylabel('Land Area Fraction (%)')
# plt.xlabel('2xCO$_{2}$ anomaly (mmd$^{-1}$)')

"""
precip 5 max
"""

axis = fig.add_subplot(224)
ax_p5_mid = axis

var = 'precip5max'

# plt.title('5 Day Max Precip (mmd$^{-1}$)')

CO2_anom, SRM_anom, masks, weights = hist2d_gfdl_data(gfdl_data, var, weight, frac_100=frac_100)

xmin = -25.
xmax = 25.
step = 1.

values = np.arange(xmin, xmax+step, step) # need to extend just beyond end to add endpoint
centres = np.arange(xmin-step/2, xmax+step, step)

plt.ylim(0,100)
plt.xlim(xmin,xmax)

total = np.array(fraction_distribution(CO2_anom, values, cumulative=False, sample_weight=weight))

better_plot = bw_off_plot_func(CO2_anom,values,masks['better'],weight)
b_nosign_plot = bw_off_plot_func(CO2_anom,values,masks['b_nosign'],weight) # better including non significant results
w_nosign_plot = bw_off_plot_func(CO2_anom,values,masks['w_nosign'],weight) # worse "" ""
worse_plot = bw_off_plot_func(CO2_anom,values,masks['worse'],weight)

# plot better
plt.fill_between(centres,100,100-b_nosign_plot,color=l_blue,lw=0)
plt.fill_between(centres,100,100-better_plot,color=blue,lw=0)
# plot worse
plt.fill_between(centres,0,w_nosign_plot,color=l_red,lw=0) # lw = 0 removes line at edges
plt.plot(centres,w_nosign_plot,color='k',lw=0.6)
plt.fill_between(centres,0,worse_plot,color=red,lw=0) # lw = 0 removes line at edges

if line_1_99_bottom:
    range_1_99 = weighted_quantile(CO2_anom, [0.01,0.99], sample_weight=weight)
    plt.axvline(x=range_1_99[0],color='0.5', lw=0.6)
    plt.axvline(x=range_1_99[1],color='0.5', lw=0.6)

# plt.xlabel('2xCO$_{2}$ anomaly (mmd$^{-1}$)')

# Create a Rectangle patch
rect = patches.Rectangle((28,18),23,82,linewidth=1,edgecolor='k',facecolor='white',clip_on=False)

# Add the patch to the Axes
ax_p5_mid.add_patch(rect)

# plt.text(29, 83, "Moderated,\nsignificant",clip_on=False, color=blue, 
#          va="baseline", ha="left", multialignment="left")
# plt.text(29, 63, "Moderated,\ninsignificant",clip_on=False, color=l_blue,
#          va="baseline", ha="left", multialignment="left")
# plt.text(29, 43, "Exacerbated,\ninsignificant",clip_on=False, color=l_red,
#          va="baseline", ha="left", multialignment="left")
# plt.text(29, 23, "Exacerbated,\nsignificant",clip_on=False, color=red,
#          va="baseline", ha="left", multialignment="left")

"""
Finish up figures
"""

fig.subplots_adjust(right=0.85)
# add_axes defines new area with: X_start, Y_start, width, height
cax = fig.add_axes([0.85,0.53,0.03,0.35])
cbar = fig.colorbar(img[3], cax=cax, ticks=bounds, format='%0.0e')
cbar.set_ticklabels([])
# cbar.set_label('Land Area Fraction')

# reduce space between ticks and tick labels
ax_pe_top.tick_params(pad=2)
ax_p5_top.tick_params(pad=2)
ax_pe_mid.tick_params(pad=2)
ax_p5_mid.tick_params(pad=2)
cax.tick_params(pad=2)

# set PE ticks
ax_pe_top.get_xaxis().set_ticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
ax_pe_mid.get_xaxis().set_ticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5]) #[-1.5,-0.75,0.0,0.75,1.5])
ax_pe_top.get_xaxis().set_ticklabels(['-1.5','-1.0','-0.5','0.0','0.5','1.0','1.5'])
ax_pe_mid.get_xaxis().set_ticklabels(['-1.5','-1.0','-0.5','0.0','0.5','1.0','1.5']) #['-1.5','-0.75','0.0','0.75','1.5'])

ax_p5_top.get_xaxis().set_ticks([-25,-12,0,12,25])
ax_p5_top.get_yaxis().set_ticks([-25,-12,0,12,25])
ax_p5_mid.get_xaxis().set_ticks([-25,-12,0,12,25])

# hide tall axis tick labels
ax_pe_top.get_xaxis().set_ticklabels([])
ax_p5_top.get_xaxis().set_ticklabels([])
ax_pe_top.get_yaxis().set_ticklabels([])
ax_p5_top.get_yaxis().set_ticklabels([])

ax_pe_mid.get_xaxis().set_ticklabels([])
ax_p5_mid.get_xaxis().set_ticklabels([])
ax_pe_mid.get_yaxis().set_ticklabels([])
ax_p5_mid.get_yaxis().set_ticklabels([])

ax_pe_mid.set_aspect(3./100.) # set so bottom panel has same aspect ratio as top
ax_p5_mid.set_aspect(50./100.)

"""
colorbar articles:
https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.colorbar
https://matplotlib.org/users/colormapnorms.html
https://matplotlib.org/tutorials/colors/colorbar_only.html
https://stackoverflow.com/questions/21952100/setting-the-limits-on-a-colorbar-in-matplotlib
"""

fig.subplots_adjust(left=0.15, right=0.83, wspace = 0.2, hspace=0.2)

plt.savefig(out_dir+'fig_2_nolabels.png', format='png', dpi=600)
plt.savefig(out_dir+'fig_2_nolabels.eps', format='eps', dpi=600)
    
plt.show()