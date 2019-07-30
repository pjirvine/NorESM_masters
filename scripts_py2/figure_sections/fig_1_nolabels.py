
#customize ticks
import matplotlib.ticker as ticker

"""
Set standard plot options
"""

hist_kwargs = {'histtype':'step', 'color':['b','r']}

weighting = 'land_noice_area'
hist_kwargs['weights'] = [gfdl_masks[weighting].flatten(), gfdl_masks[weighting].flatten()]

"""
Define plot data
"""

def plot_data(var): # For main figure
    # use GFDL variable names
    Geo_anom = gfdl_data['SRM',var,'mean'] - gfdl_data['ctrl',var,'mean']
    CO2_anom = gfdl_data['CO2',var,'mean'] - gfdl_data['ctrl',var,'mean']
    return [Geo_anom.flatten(), CO2_anom.flatten()]

def plot_data_100(var): # For suppl figure with scaled up results
    # use GFDL variable names
    CO2_glbl_sat_anom = abs(np.sum((gfdl_data['CO2','tas','mean'] - gfdl_data['ctrl','tas','mean']) * gfdl_masks['area']))
    SRM_CO2_glbl_sat_anom = abs(np.sum((gfdl_data['SRM','tas','mean'] - gfdl_data['CO2','tas','mean']) * gfdl_masks['area']))
    
    anom_ratio = CO2_glbl_sat_anom / SRM_CO2_glbl_sat_anom
    
    new_SRM = gfdl_data['CO2',var,'mean'] + anom_ratio * (gfdl_data['SRM',var,'mean'] - gfdl_data['CO2',var,'mean'])
    
    Geo_anom = new_SRM - gfdl_data['ctrl',var,'mean']
    CO2_anom = gfdl_data['CO2',var,'mean'] - gfdl_data['ctrl',var,'mean']
    return [Geo_anom.flatten(), CO2_anom.flatten()]

def plot_data_pc(var): # For suppl figure
    # use GFDL variable names
    Geo_anom = 100. * ((gfdl_data['SRM',var,'mean'] / gfdl_data['ctrl',var,'mean']) - 1.0)
    CO2_anom = 100. * ((gfdl_data['CO2',var,'mean'] / gfdl_data['ctrl',var,'mean']) - 1.0)
    return [Geo_anom.flatten(), CO2_anom.flatten()]

def plot_data_SD(var): # For suppl figure
    # use GFDL variable names
    Geo_anom = (gfdl_data['SRM',var,'mean'] - gfdl_data['ctrl',var,'mean']) / gfdl_data['ctrl',var,'std']
    CO2_anom = (gfdl_data['CO2',var,'mean'] - gfdl_data['ctrl',var,'mean']) / gfdl_data['ctrl',var,'std']
    return [Geo_anom.flatten(), CO2_anom.flatten()]

def boxplot_2(axis,CO2_land,CO2_pop,SRM_land,SRM_pop,labels=False):

    def box_rectangles(axis, quantiles, y_loc, thick, color):
        
        thin = thick*0.5
        thinner = thick*0.2
        
        # create a rectangle
        patches = [
            # 1-99% range
            mpatches.Rectangle((quantiles[0],y_loc-0.5*thinner), quantiles[-1] - quantiles[0], thinner, facecolor=color, linewidth=0), ### Background
            # 5-95% range
            mpatches.Rectangle((quantiles[1],y_loc-0.5*thin), quantiles[-2] - quantiles[1], thin, facecolor=color, linewidth=0), ### Background
            # 25-75% range
            mpatches.Rectangle((quantiles[2],y_loc-0.5*thick), quantiles[-3] - quantiles[2], thick, facecolor=color, linewidth=0), ### Background
        ]
        for p in patches:
            axis.add_patch(p)
            
        axis.plot([quantiles[3],quantiles[3]],[y_loc-0.5*thick,y_loc+0.5*thick],'w',linewidth=1)
    #end def
    
    # set y locations for bars
    y_CO2_land, y_SRM_land = 0.16, 0.34
    y_CO2_pop, y_SRM_pop = 0.66, 0.84

    # set basic thickness
    thick = 0.15
    
    axis.set_ylim(0,1)
    axis.yaxis.set_major_locator(ticker.NullLocator())
    
    axis.plot([0,0],[0,1],'k',linewidth=1,zorder=0)
    axis.axhline(0.5,color='grey',linewidth=0.6)
    
    # plot the shapes:
    box_rectangles(axis, CO2_land, y_CO2_land, thick, red)
    box_rectangles(axis, SRM_land, y_SRM_land, thick, blue)
    box_rectangles(axis, CO2_pop, y_CO2_pop, thick, red)
    box_rectangles(axis, SRM_pop, y_SRM_pop, thick, blue)
#end def
    
    
"""
Figure settings
"""

out_dir = '/n/home03/pjirvine/projects/fraction_better_off/figures/fig_1/'

weighting = 'land_noice_area' # old
weight = gfdl_masks[weighting].flatten() # old

land_weight = gfdl_masks['land_noice_area'].flatten()
pop_weight = gfdl_masks['pop'].flatten()

quantiles = [0.01,0.05,0.25,0.5,0.75,0.95,0.99]

fig = plt.figure(figsize=cm2inch(8.5,14))

"""
SAT plot
"""

ax = fig.add_subplot(411)

# get plot data together and weight
data = plot_data('tas')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels and title
unit = '$^\circ$C'
# plt.xlabel('T anomaly ({unit})'.format(unit=unit))
plt.xlim(0,5)
ax.xaxis.set_ticklabels([])

# plt.text(3.5,0.35, "land area", ha='left',va='center')
# plt.text(3.5,0.65, "population", ha='left',va='center')

"""
SAT max plot
"""

ax = fig.add_subplot(412)

# get plot data together and weight
data = plot_data('tasmax')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels and title
unit = '$^\circ$C'
# plt.xlabel('Tx anomaly ({unit})'.format(unit=unit))
plt.xlim(0,5)
ax.xaxis.set_ticklabels([])

"""
P -E plot
"""

ax = fig.add_subplot(413)

# get plot data together and weight
data = plot_data('pe')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels
unit = 'mmDay$^{-1}$'
# plt.xlabel('PE anomaly ({unit})'.format(unit=unit))
plt.xlim(-0.8,0.8)
plt.xticks([-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8])
ax.xaxis.set_ticklabels([])

"""
Preicp 5 day max plot
"""
                 
ax = fig.add_subplot(414)
                 
# get plot data together and weight
data = plot_data('precip5max')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels
unit = 'mmDay$^{-1}$'
# plt.xlabel('Px anomaly ({unit})'.format(unit=unit))
plt.xlim(-10,20)
ax.xaxis.set_ticklabels([])

"""
Figure finalizing
"""

plt.subplots_adjust(top=0.98, bottom=0.1, left=0.10, right=0.95, hspace=0.8,
                    wspace=0.35)

plt.savefig(out_dir+'fig1_nolabels.png', format='png', dpi=480)
plt.savefig(out_dir+'fig1_nolabels.eps', format='eps', dpi=480)

plt.show()

"""
###
###
END main FIGURE
###
###
"""