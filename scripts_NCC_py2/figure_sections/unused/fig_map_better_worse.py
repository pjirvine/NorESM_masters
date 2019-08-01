# Load modules

import cfplot as cfp
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import mpl_toolkits.basemap as basemap

# from matplotlib import cm

"""
Gather and regrid all data
"""

ref_dir = '/n/home03/pjirvine/keithfs1_pji/model_ref_files/netcdf/'

# Standard inputs for simulations
get_data_args = [model_exp_runs, 'ann']
get_data_kwargs = {'time':'11-50', 'nyears':40, 'flatten':False}

ttest = 0.1

# Regrid to this super-hi HiFLOR resolution.
out_ncfile = ref_dir + 'HiFLOR.nc'

# Dictionaries to hold each vars results for each model

abs_diff_vars = {} 
better_vars = {}

# var_list = ['tas_Amon','txxETCCDI_yr','pr_Amon','rx5dayETCCDI_yr','p-e_Amon']

for var in var_list: # loop over var_list
    
    print var
    
    # Dictionaries to hold each models results
    
    abs_diff_dict = {}
    better_dict = {}

    for model in model_list: # Loop over model_list
        
        print model

        if 'ETCCDI' in var:
            in_ncfile = ref_dir + 'NorESM1-M' + '.nc'
        else:
            in_ncfile = ref_dir + model + '.nc'

        var_filename = var
        if model != 'NorESM1-M':
            var_filename = var + var_name_mod[var]

        # Get all metrics for this model / var
        model_data = get_data_dict(var_filename, model, *get_data_args, 
                                   var_offset=var_offsets[var], var_mult=var_mults[var], fraction=fraction, 
                                   ttest_level=ttest, **get_data_kwargs)

        # Only calculated values if model_data is a dict / contains results
        if type(model_data) is dict:

            # abs_diff results:
            in_arr = abs(model_data['CO2_effect']) - abs(model_data['frac_srm_CO2_effect'])
            out_arr = cdo_regrid_array(in_ncfile, in_arr, out_ncfile)
            abs_diff_dict[model] = np.squeeze(out_arr)

            # Better / Worse Off results:
            better_worse_off = 1. * model_data['better_off'] -1. * model_data['worse_off']
            out_arr = cdo_regrid_array(in_ncfile, better_worse_off, out_ncfile)
            better_dict[model] = np.squeeze(out_arr)
    
    # End of model loop
        
    abs_diff_vars[var] = abs_diff_dict
    better_vars[var] = better_dict

# End of var loop

"""
Sum / mean over models
"""

bwoff_sum = {}
mean_abs_diff = {}

for var in var_list:

    bwoff_sum[var,'total'] = sum(better_vars[var].values())
    
    better_list = [1. * (X > 0) for X in better_vars[var].values()]
    bwoff_sum[var,'better'] = sum(better_list)
    
    worse_list = [1. * (X < 0) for X in better_vars[var].values()]
    bwoff_sum[var,'worse'] = sum(worse_list)
    
    mean_abs_diff[var] = sum(abs_diff_vars[var].values()) / len(abs_diff_dict)


    
    
"""
SPLIT HERE
""" 



    
def better_off_map(lon, lat, data, levels, ticks, color=plt.cm.Blues, 
                   title='', bottom_labels=True, left_labels=True, 
                   ocean=True, fix_aspect=False):

    plt.title(title)
    
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
                llcrnrlon=-180,urcrnrlon=180,resolution='c',fix_aspect=fix_aspect)

    """
    Create shifted lat-lon grid and set plot data
    """

    #shift data and lats/lons
    data_shift, lon_shift = basemap.shiftgrid(180.,data,lon)
    lon_shift = lon_shift -360. # needed as shiftgrid is a pain

    # Create X-Y grid for map
    lon2, lat2 = np.meshgrid(lon_shift,lat)
    x, y = m(lon2, lat2)

    # plot data = shifted and ocean-masked (if True)
    if ocean:
        data_plot = basemap.maskoceans(lon2,lat2,data_shift)
    else:
        data_plot = data_shift

    """
    Setup map
    """

    m.drawcoastlines()
    m.drawmapboundary(fill_color='white')
    if left_labels:
        m.drawparallels(np.arange(-90,120,30),labels=[1,0,0,0])
    else:
        m.drawparallels(np.arange(-90,120,30))
    if bottom_labels:
        m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,60),labels=[0,0,0,1])
    else:
        m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,60))

    """
    Plot contours
    """

    cs = m.contourf(x,y,data_plot,cmap=color,levels=levels)

    plt.colorbar(ticks=ticks)#, label='Number of Models')

"""
Set up common plot settings
"""

out_dir = '/n/home03/pjirvine/projects/fraction_better_off/figures/fig_3/'

ocean = False

file_out = 'fig_3_ocean'

# load lat, lons
hiflor = Dataset(out_ncfile)
lon = hiflor.variables['lon'][:]
lat = hiflor.variables['lat'][:]

# Levels for 12
levels_12 = np.arange(0.5,13.5,1) # 12 models for other vars.
ticks_12 = np.arange(1,14,1)  

# Levels for 8 (extreme indices)
levels_8 = np.arange(0.5,9.5,1) # Only 8 models have extreme data
ticks_8 = np.arange(1,9,1)

plt.rcParams.update({'font.size': 8})

"""
Start figure
"""

fig = plt.figure(figsize=cm2inch((19,19)))

"""
Precip
"""

axis = fig.add_subplot(321)

title = 'Precip - Models Better Off'

var = 'pr_Amon'
data = bwoff_sum[var,'better']

better_off_map(lon, lat, data, levels_12, ticks_12, color=plt.cm.Blues, title=title, bottom_labels=False, ocean=ocean, fix_aspect=False)

axis = fig.add_subplot(322)

title = 'Precip - Models Worse Off'

var = 'pr_Amon'
data = bwoff_sum[var,'worse']

better_off_map(lon, lat, data, levels_12, ticks_12, color=plt.cm.Reds, title=title, bottom_labels=False, left_labels=False, ocean=ocean, fix_aspect=False)

"""
Precip - Evap
"""

axis = fig.add_subplot(323)

title = 'P-E - Models Better Off'

var = 'p-e_Amon'
data = bwoff_sum[var,'better']

better_off_map(lon, lat, data, levels_12, ticks_12, color=plt.cm.Blues, title=title, bottom_labels=False, ocean=ocean, fix_aspect=False)

axis = fig.add_subplot(324)

title = 'P-E - Models Worse Off'

var = 'p-e_Amon'
data = bwoff_sum[var,'worse'] 

better_off_map(lon, lat, data, levels_12, ticks_12, color=plt.cm.Reds, title=title, bottom_labels=False, left_labels=False, ocean=ocean, fix_aspect=False)

"""
Precip 5day max
"""

axis = fig.add_subplot(325)

title = 'P5day - Models Better Off'

var = 'rx5dayETCCDI_yr'
data = bwoff_sum[var,'better']

better_off_map(lon, lat, data, levels_8, ticks_8, color=plt.cm.Blues, title=title, ocean=ocean, fix_aspect=False)

axis = fig.add_subplot(326)

title = 'P5day - Models Worse Off'

var = 'rx5dayETCCDI_yr'
data = bwoff_sum[var,'worse']

better_off_map(lon, lat, data, levels_8, ticks_8, color=plt.cm.Reds, title=title, left_labels=False, ocean=ocean, fix_aspect=False)


"""
Finalize figure
"""

fig.subplots_adjust(wspace=0)

plt.savefig(out_dir+file_out+'.png', format='png', dpi=480)
plt.savefig(out_dir+file_out+'.eps', format='eps', dpi=480)
plt.show()