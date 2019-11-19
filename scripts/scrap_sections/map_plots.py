"""
MAP PLOTS
"""

"""
Anom Maps
"""

out_dir = '/n/home03/pjirvine/projects/GLENS_fraction_better_off/plots/'

"""
Import modules
"""

# IMPORT MODULES
import matplotlib.pylab as plt
%matplotlib inline
import numpy as np
from matplotlib import cm
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

def anom_map(data, bounds, labels, title, cbar_label, out_name=None, show=False):

    # Make CO2 anom figure
    fig = plt.figure(figsize=(13,6.2))

    plt.rcParams.update({'font.size': 16})

    ax = plt.subplot(111, projection=ccrs.PlateCarree())

    data_cyc, lons_cyc = add_cyclic_point(data, coord=lons)
    lons2d_cyc, lats2d = np.meshgrid(lons_cyc, lats)

    mm = ax.pcolormesh(lons2d_cyc, lats2d, data_cyc, vmin=bounds[0], vmax=bounds[-1],
                       transform=ccrs.PlateCarree(),cmap='RdBu' )
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
Anom map set function
"""

def anom_map_set(all_data, var, var_title, anom_type, cbar_label, bounds, labels, cmap=None):
    cases = ['Full-GLENS','RCP8.5','Baseline']
    sg_anom, CO2_anom, sg_CO2_anom, masks, weights, fractions = better_worse_full_data(all_data, cases[0], cases[1], cases[2], var, weight, nyears=80, ttest_level=0.1, anom_type=anom_type)
    # get data for HALF-GLENS
    cases = ['Half-GLENS','RCP8.5','Baseline']
    half_sg_anom, CO2_anom, half_sg_CO2_anom, masks, weights, fractions = better_worse_full_data(all_data, cases[0], cases[1], cases[2], var, weight, nyears=80, ttest_level=0.1, anom_type=anom_type)

    # Produce plots
    title = 'RCP8.5 - Baseline, ' + var_title
    out_name = 'RCP8.5_'+var+'_'+anom_type+'_anom'
    anom_map(CO2_anom.transpose(), bounds, labels, title, cbar_label, out_name=out_name, show=True)

    title = 'Full-GLENS - Baseline, ' + var_title
    out_name = 'Full-GLENS_'+var+'_'+anom_type+'_anom'
    anom_map(sg_anom.transpose(), bounds, labels, title, cbar_label, out_name=out_name, show=True)

    # Note this is inverted!
    title = 'RCP8.5 - Full-GLENS, ' + var_title
    out_name = 'RCP8.5-Full-GLENS_'+var+'_'+anom_type+'_anom'
    anom_map(-1. * sg_CO2_anom.transpose(), bounds, labels, title, cbar_label, out_name=out_name, show=True)

    title = 'Half-GLENS - Baseline, ' + var_title
    out_name = 'Half-GLENS_'+var+'_'+anom_type+'_anom'
    anom_map(half_sg_anom.transpose(), bounds, labels, title, cbar_label, out_name=out_name, show=True)
#end def

"""
Produce Anom Maps
"""

out_dir = '/n/home03/pjirvine/projects/GLENS_fraction_better_off/plots/'

weight = all_masks['area']

var = 'P-E'
var_title = 'Precipitation - Evaporation'
anom_type = 'SD'
cbar_label = 'Baseline STDs'
bounds = [-1,-0.5,0,0.5,1.0]
labels = ["{:2.1f}".format(IDX) for IDX in bounds]
cmap = None

anom_map_set(all_data, var, var_title, anom_type, cbar_label, bounds, labels, cmap=None)
