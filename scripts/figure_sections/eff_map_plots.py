"""
Efficacy map function
"""

def eff_map(data, bounds, labels, title, cbar_label, mask, mask_threshold=0.27, out_name=None, show=False):
    """
    mask threshold: 0.27 = 90% T-Test, 0.32 = 95% T-Test
    """

    """
    Define new colormap
    """

    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    rdylgn_1 = cm.get_cmap('RdYlGn', 200)
    rdylgn_2 = cm.get_cmap('RdYlGn', 200)
    rdbu = cm.get_cmap('PRGn', 200)

    color_a = rdylgn_1(np.linspace(0, 1, 200))
    color_b = rdylgn_2(np.linspace(0, 1, 200))
    color_c = rdbu(np.flip(np.linspace(0, 1, 200)))

    color_list = [color_b[-1], (0.8, 0.8, 1), color_c[-1]]
    cmap_from_list = LinearSegmentedColormap.from_list('my_colors', color_list, N=200)
    color_d = cmap_from_list(np.linspace(0, 1, 200))

    newcolors = np.vstack((color_a[0:100], color_b[100:200], color_d))
    newcmp = ListedColormap(newcolors, name='spectral_pete')

    """
    make plot
    """

    fig = plt.figure(figsize=(13,6.2))

    plt.rcParams.update({'font.size': 16})

    ax = plt.subplot(111, projection=ccrs.PlateCarree())

    data_cyc, lons_cyc = add_cyclic_point(data, coord=lons)
    mask_cyc, lons_cyc = add_cyclic_point(mask, coord=lons)
    lons2d_cyc, lats2d = np.meshgrid(lons_cyc, lats)

    masked_data = np.ma.array(data_cyc,mask=abs(mask_cyc) < mask_threshold)

    mm = ax.pcolormesh(lons2d_cyc, lats2d, masked_data, vmin=bounds[0], vmax=bounds[-1],
                       transform=ccrs.PlateCarree(),cmap=newcmp )
    ax.coastlines()

    plt.title(title)

    cbar = fig.colorbar(mm, ax=ax, ticks=bounds)
    cbar.set_ticklabels(labels)
    cbar.set_label(cbar_label)

    # fig.subplots_adjust(right=0.85)
    # # add_axes defines new area with: X_start, Y_start, width, height
    # cax = fig.add_axes([0.85,0.53,0.03,0.35])

    if out_name is not None:
        plt.savefig(out_dir+out_name+'.png', format='png', dpi=600)
        plt.savefig(out_dir+out_name+'.eps', format='eps', dpi=600)

    if show:
        plt.show()
#end def   
"""
Efficacy Map example
"""

# Set cases, var and get data
cases = ['Full-GLENS','RCP8.5','Baseline']
var = 'PRECTMX'
weight = all_masks['area']
# get SD anoms
sg_anom, CO2_anom, sg_CO2_anom, masks, weights, fractions = better_worse_full_data(all_data, cases[0], cases[1], cases[2], var, weight, nyears=80, ttest_level=0.1, anom_type='SD')

eff = -1. * (sg_CO2_anom / CO2_anom)

data = eff.transpose()
mask = CO2_anom.transpose()

bounds = [-1, -0.5, 0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
labels = [str(IDX) for IDX in bounds]
title = var+' GLENS Efficacy'
cbar_label = 'Efficacy (ratio)'
out_name = var+'_eff_map' 

eff_map(data, bounds, labels, title, cbar_label, mask, mask_threshold=0.27, out_name=out_name, show=False)
