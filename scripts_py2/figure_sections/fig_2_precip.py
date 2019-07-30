
"""
Load and run fig_2.py first
"""

"""
settings
"""

out_dir = '/n/home03/pjirvine/projects/fraction_better_off/figures/fig_2/'

frac_100 = False

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
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, clip=True)

# g_cmap = plt.cm.gist_gray
# g_norm = mpl.colors.BoundaryNorm(bounds, g_cmap.N)

nbins = 200

"""
Create figure
"""

fig = plt.figure(figsize=cm2inch(8,13))
plt.rcParams.update({'font.size': 8})

"""
Start TOP
"""

"""
p-e no filter
"""

axis = fig.add_subplot(211)
ax_precip_top = axis

var = 'precip'

plt.title('Precip')

plt.axis('scaled')

CO2_anom, SRM_anom, masks, weights = hist2d_gfdl_data(gfdl_data, var, weight, frac_100=frac_100)

# Set axes and add lines

xlims = [-1.5,1.5]
ylims = xlims
xnum_steps, ynum_steps = 7, 7

sort_axes(axis, xlims, ylims, xnum_steps, ynum_steps, num_format='%0.1f')

add_lines(axis)

# produce plot

img = axis.hist2d(CO2_anom, SRM_anom, bins=nbins, range = [xlims,ylims], weights=weight, norm=norm, cmap=cmap, cmin=1.e-12)
# img = ax_pe.hist2d(CO2_anom[certain], SRM_anom[certain], bins=100, range = [xlims,ylims], weights=weight[certain], norm=norm, cmap=cmap, cmin=1.e-12)

range_1_99 = weighted_quantile(CO2_anom, [0.01,0.99], sample_weight=weight)
plt.axvline(x=range_1_99[0],color='0.5', lw=0.6)
plt.axvline(x=range_1_99[1],color='0.5', lw=0.6)

if frac_100:
    plt.ylabel('Full-SG - CTRL (mmd$^{-1}$)')
else:
    plt.ylabel('Half-SG - CTRL (mmd$^{-1}$)')

"""
BOTTOM FIGURE

precip no filter
"""

axis = fig.add_subplot(212, sharex=ax_precip_top)
ax_precip_bottom = axis

var = 'precip'

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
plt.fill_between(centres,100,100-b_nosign_plot,color=blue,alpha=std_alpha,lw=0)
plt.fill_between(centres,100,100-better_plot,color=blue,lw=0)
# plot worse
plt.fill_between(centres,0,w_nosign_plot,color=red,alpha=std_alpha,lw=0) # lw = 0 removes line at edges
plt.fill_between(centres,0,worse_plot,color=red,lw=0) # lw = 0 removes line at edges

range_1_99 = weighted_quantile(CO2_anom, [0.01,0.99], sample_weight=weight)
plt.axvline(x=range_1_99[0],color='0.5', lw=0.6)
plt.axvline(x=range_1_99[1],color='0.5', lw=0.6)

plt.ylabel('Fraction (%)')
plt.xlabel('2xCO$_{2}$ - CTRL (mmd$^{-1}$)')

"""
Finish up figures
"""

# fig.subplots_adjust(right=0.8)
# add_axes defines new area with: X_start, Y_start, width, height
cax = fig.add_axes([0.8,0.5,0.06,0.39])
cbar = fig.colorbar(img[3], cax=cax, ticks=bounds, extend='both', format='%0.0e')
cbar.set_ticklabels(labels)

# reduce space between ticks and tick labels
ax_precip_top.tick_params(pad=2)
ax_precip_bottom.tick_params(pad=2)
cax.tick_params(pad=2)

# set PE ticks
ax_precip_top.get_xaxis().set_ticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
ax_precip_bottom.get_xaxis().set_ticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5]) 
ax_precip_top.get_xaxis().set_ticklabels(['-1.5','-1.0','-0.5','0.0','0.5','1.0','1.5'])
ax_precip_bottom.get_xaxis().set_ticklabels(['-1.5','-1.0','-0.5','0.0','0.5','1.0','1.5']) 

# hide top x axis tick labels
ax_precip_top.get_xaxis().set_ticklabels([])

"""
colorbar articles:
https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.colorbar
https://matplotlib.org/users/colormapnorms.html
https://matplotlib.org/tutorials/colors/colorbar_only.html
https://stackoverflow.com/questions/21952100/setting-the-limits-on-a-colorbar-in-matplotlib
"""

fig.subplots_adjust(left=0.15, right=0.75, wspace = 0.2, hspace=0.1)

plt.savefig(out_dir+'fig_2_precip.png', format='png', dpi=480)
plt.savefig(out_dir+'fig_2_precip.eps', format='eps', dpi=480)
    
plt.show()