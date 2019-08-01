
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
plt.xlabel('T anomaly ({unit})'.format(unit=unit))
plt.xlim(0,5)

plt.text(3.5,0.35, "land area", ha='left',va='center')
plt.text(3.5,0.65, "population", ha='left',va='center')

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
plt.xlabel('Tx anomaly ({unit})'.format(unit=unit))
plt.xlim(0,5)

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
plt.xlabel('PE anomaly ({unit})'.format(unit=unit))
plt.xlim(-0.8,0.8)
plt.xticks([-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8])

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
plt.xlabel('Px anomaly ({unit})'.format(unit=unit))
plt.xlim(-10,20)

"""
Figure finalizing
"""

plt.subplots_adjust(top=0.98, bottom=0.1, left=0.10, right=0.95, hspace=0.8,
                    wspace=0.35)

plt.savefig(out_dir+'fig1.png', format='png', dpi=480)
plt.savefig(out_dir+'fig1.eps', format='eps', dpi=480)

plt.show()

"""
###
###
END main FIGURE
###
###
"""


"""
Supplementary GFDL scales to 100% offset
"""

fig = plt.figure(figsize=cm2inch(8.5,14))

"""
SAT plot - 100%
"""

ax = fig.add_subplot(411)

# get plot data together and weight
data = plot_data_100('tas')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels and title
unit = '$^\circ$C'
plt.xlabel('T anomaly ({unit})'.format(unit=unit))
plt.xlim(-2,5)

"""
SAT max plot - 100%
"""

ax = fig.add_subplot(412)

# get plot data together and weight
data = plot_data_100('tasmax')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels and title
unit = '$^\circ$C'
plt.xlabel('Tx anomaly ({unit})'.format(unit=unit))
plt.xlim(-2,5)

"""
P -E plot - 100%
"""

ax = fig.add_subplot(413)

# get plot data together and weight
data = plot_data_100('pe')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels
unit = 'mmDay$^{-1}$'
plt.xlabel('PE anomaly ({unit})'.format(unit=unit))
plt.xlim(-0.8,0.8)

"""
Preicp 5 day max plot - 100%
"""
                 
ax = fig.add_subplot(414)
                 
# get plot data together and weight
data = plot_data_100('precip5max')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels
unit = 'mmDay$^{-1}$'
plt.xlabel('Px anomaly ({unit})'.format(unit=unit))
plt.xlim(-20,20)

plt.text(-18,0.25, "land area", ha='left',va='center')
plt.text(-18,0.75, "population", ha='left',va='center')

"""
Figure finalizing
"""

plt.subplots_adjust(top=0.98, bottom=0.1, left=0.10, right=0.95, hspace=0.8,
                    wspace=0.35)

plt.savefig(out_dir+'fig1_100.png', format='png', dpi=480)
plt.savefig(out_dir+'fig1_100.eps', format='eps', dpi=480)

plt.show()

"""
###
###
END 100% figure
###
###
"""




"""
Supplementary % and SD change version
"""

fig = plt.figure(figsize=cm2inch(8.5,14))

"""
% P -E plot 
"""

ax = fig.add_subplot(411)

# get plot data together and weight
data = plot_data_pc('pe')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels
unit = '%'
plt.xlabel('PE anomaly ({unit})'.format(unit=unit))
plt.xlim(-200,200)

"""
% Preicp 5 day max plot 
"""
                 
ax = fig.add_subplot(413)
                 
# get plot data together and weight
data = plot_data_pc('precip5max')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels
unit = '%'
plt.xlabel('Px anomaly ({unit})'.format(unit=unit))
plt.xlim(-50,50)

"""
SD P -E plot 
"""

ax = fig.add_subplot(412)

# get plot data together and weight
data = plot_data_SD('pe')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels
unit = 'Control SDs'
plt.xlabel('PE anomaly ({unit})'.format(unit=unit))
plt.xlim(-2,2)

"""
SD Preicp 5 day max plot 
"""
                 
ax = fig.add_subplot(414)
                 
# get plot data together and weight
data = plot_data_SD('precip5max')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels
unit = 'Control SDs'
plt.xlabel('Px anomaly ({unit})'.format(unit=unit))
plt.xlim(-2,2)

plt.text(-1.8,0.25, "land area", ha='left',va='center')
plt.text(-1.8,0.75, "population", ha='left',va='center')

"""
Figure finalizing
"""

plt.subplots_adjust(top=0.98, bottom=0.1, left=0.10, right=0.95, hspace=0.8,
                    wspace=0.35)

plt.savefig(out_dir+'fig1_pc_sd.png', format='png', dpi=480)
plt.savefig(out_dir+'fig1_pc_sd.eps', format='eps', dpi=480)

plt.show()


"""
###
###
End of fig 1 SD / %
###
###
"""


"""
Supplementary fig 1 PRECIP mm/d, % and SD change version
"""

fig = plt.figure(figsize=cm2inch(8.5,10.5))

"""
Precip mm/d plot
"""

ax = fig.add_subplot(311)

# get plot data together and weight
data = plot_data('precip')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels
unit = 'mmDay$^{-1}$'
plt.xlabel('P anomaly ({unit})'.format(unit=unit))
plt.xlim(-1.0,1.0)

"""
Precip % plot 
"""

ax = fig.add_subplot(312)

# get plot data together and weight
data = plot_data_pc('precip')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels
unit = '%'
plt.xlabel('P anomaly ({unit})'.format(unit=unit))
plt.xlim(-50,50)

"""
SD Precip plot 
"""

ax = fig.add_subplot(313)

# get plot data together and weight
data = plot_data_SD('precip')
CO2_land = weighted_quantile(data[1], quantiles, sample_weight=land_weight)
SRM_land = weighted_quantile(data[0], quantiles, sample_weight=land_weight)
CO2_pop = weighted_quantile(data[1], quantiles, sample_weight=pop_weight)
SRM_pop = weighted_quantile(data[0], quantiles, sample_weight=pop_weight)

boxplot_2(ax, CO2_land, CO2_pop, SRM_land, SRM_pop)

# set axes labels
unit = 'Control SDs'
plt.xlabel('P anomaly ({unit})'.format(unit=unit))
plt.xlim(-2.5,2.5)

plt.text(-2.4,0.25, "land area", ha='left',va='center')
plt.text(-2.4,0.75, "population", ha='left',va='center')

"""
Figure finalizing
"""

plt.subplots_adjust(top=0.98, bottom=0.1, left=0.10, right=0.95, hspace=0.8,
                    wspace=0.35)

plt.savefig(out_dir+'fig1_precip.png', format='png', dpi=480)
plt.savefig(out_dir+'fig1_precip.eps', format='eps', dpi=480)

plt.show()


"""
###
###
End of fig 1 Precip mm/d, SD  %
###
###
"""

"""
Table output from fig 1
"""

table_dir = '/n/home03/pjirvine/projects/fraction_better_off/tables/'

land_quantiles = {}

for var in vars_hiflor:
    
    quantile_dict = {}
    
    data = plot_data(var)
    
    for quantile in quantiles:
        
        quantile_dict['CO2_{:.2f}'.format(quantile)] = weighted_quantile(data[1], quantile, sample_weight=land_weight)
        quantile_dict['SRM_{:.2f}'.format(quantile)] = weighted_quantile(data[0], quantile, sample_weight=land_weight)
    
    land_quantiles[var] = quantile_dict       
    
land_out = pd.DataFrame.from_dict(land_quantiles).to_csv(table_dir+'fig1_land_quantiles.csv')