
#customize ticks
import matplotlib.ticker as ticker

"""
Set standard plot options
"""

hist_kwargs = {'histtype':'step', 'color':['b','r']}

weighting = 'land_noice_area'
hist_kwargs['weights'] = [all_masks[weighting].flatten(), all_masks[weighting].flatten()]

"""
Figure settings
"""

out_dir = '/n/home03/pjirvine/projects/GLENS_fraction_better_off/figures/'

quantiles = [0.01,0.05,0.25,0.5,0.75,0.95,0.99]

"""
CASES:
'Baseline'     - RCP8.5 @ 2010-2029
'RCP8.5'       - RCP8.5 @ 2075-2094
'Full-GLENS'   - GLENS  @ 2075-2094
'Half-GLENS'   - Scaled Half-GLENS  @ 2075-2094
### NOT DONE ### 'Baseline-2'   - Shifted Half-GLENS @ 2075-2094
### NOT DONE ### 'Half-GLENS-2' - RCP8.5 @ 2010-2029 W/ alternate runs
"""

def plot_data(var, case_a, case_b, weight_name, quantiles = [0.01,0.05,0.25,0.5,0.75,0.95,0.99]):
    # returns weighted quantiles of the anomaly for plotting
    # take anomaly of means [0] element
    anom = all_data[(var, case_a)][0] - all_data[(var, case_b)][0]
    anom_flat = anom.flatten()
    weight_flat = all_masks[weight_name].flatten()
    # return weighted quantiles of distribution
    return weighted_quantile(anom_flat, quantiles, sample_weight=weight_flat)

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

def boxplot_3(axis,bottom,mid,top,labels=False):
    """
    New GLENS style
    """
    
    # set y locations for bars
    y_bottom, y_mid, y_top = 0.2, 0.5, 0.8

    # set basic thickness
    thick = 0.2
    
    axis.set_ylim(0,1)
    axis.yaxis.set_major_locator(ticker.NullLocator())
    
    axis.plot([0,0],[0,1],'k',linewidth=1,zorder=0)
    
    # plot the shapes:
    box_rectangles(axis, bottom, y_bottom, thick, red)
    box_rectangles(axis, mid, y_mid, thick, purple)
    box_rectangles(axis, top, y_top, thick, blue)
#end def

def boxplot_2(axis,CO2_land,CO2_pop,SRM_land,SRM_pop,labels=False):
    """
    Original figure 1 style
    """
    
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
##################################
#
FIGURE 1 GLENS
#
##################################
"""

fig = plt.figure(figsize=cm2inch(8.5,14))

def fig1_land(var):
    # get and then plot data
    RCP85_land = plot_data(var, 'RCP8.5', 'Baseline', 'land_noice_area')
    HALF_GLENS_land = plot_data(var, 'Half-GLENS', 'Baseline', 'land_noice_area')
    FULL_GLENS_land = plot_data(var, 'Full-GLENS', 'Baseline', 'land_noice_area')
    
    boxplot_3(ax, RCP85_land, HALF_GLENS_land, FULL_GLENS_land)

#TREFHT plot
ax = fig.add_subplot(411)

# plot data!
var = 'TREFHT'
fig1_land(var)
    
# set axes labels and title
unit = '$^\circ$C'
plt.xlabel('T anomaly ({unit})'.format(unit=unit))
plt.xlim(-5,15)

# plt.text(3.5,0.35, "land area", ha='left',va='center')
# plt.text(3.5,0.65, "population", ha='left',va='center')


#TREFHTMX plot
ax = fig.add_subplot(412)

# plot data!
var = 'TREFHTMX'
fig1_land(var)
    
# set axes labels and title
unit = '$^\circ$C'
plt.xlabel('Tmax anomaly ({unit})'.format(unit=unit))
plt.xlim(-5,15)


#P-E plot
ax = fig.add_subplot(413)

# plot data!
var = 'P-E'
fig1_land(var)
    
# set axes labels and title
unit = 'mmDay$^{-1}$'
plt.xlabel('P-E anomaly ({unit})'.format(unit=unit))
plt.xlim(-2.5,2.5)


#PRECTMX plot
ax = fig.add_subplot(414)

# plot data!
var = 'PRECTMX'
fig1_land(var)
    
# set axes labels and title
unit = 'mmDay$^{-1}$'
plt.xlabel('Pmax anomaly ({unit})'.format(unit=unit))
plt.xlim(-160,160)


"""
Figure finalizing
"""

plt.subplots_adjust(top=0.98, bottom=0.1, left=0.10, right=0.95, hspace=0.8,
                    wspace=0.35)

plt.savefig(out_dir+'fig1.png', format='png', dpi=480)
plt.savefig(out_dir+'fig1.eps', format='eps', dpi=480)

plt.show()

"""
##################################
#
FIGURE 1 OLD STYLE
#
##################################
"""

# fig = plt.figure(figsize=cm2inch(8.5,14))

# def fig1_land_pop(var):
#     # get plot data
#     RCP85_land = plot_data(var, 'RCP8.5', 'Baseline', 'land_noice_area')
#     GLENS_land = plot_data(var, 'Full-GLENS', 'Baseline', 'land_noice_area')
#     RCP85_pop = plot_data(var, 'RCP8.5', 'Baseline', 'pop')
#     GLENS_pop = plot_data(var, 'Full-GLENS', 'Baseline', 'pop')

#     print(max(RCP85_land), min(RCP85_land))
    
#     boxplot_2(ax, RCP85_land, RCP85_pop, GLENS_land, GLENS_pop)

# #TREFHT plot
# ax = fig.add_subplot(411)

# var = 'TREFHT'

# fig1_land_pop(var)
    
# # set axes labels and title
# unit = '$^\circ$C'
# plt.xlabel('T anomaly ({unit})'.format(unit=unit))
# plt.xlim(-5,15)

# plt.text(3.5,0.35, "land area", ha='left',va='center')
# plt.text(3.5,0.65, "population", ha='left',va='center')


# #TREFHTMX plot
# ax = fig.add_subplot(412)

# var = 'TREFHTMX'

# fig1_land_pop(var)
    
# # set axes labels and title
# unit = '$^\circ$C'
# plt.xlabel('Tmax anomaly ({unit})'.format(unit=unit))
# plt.xlim(-5,15)


# #P-E plot
# ax = fig.add_subplot(413)

# var = 'P-E'

# fig1_land_pop(var)
    
# # set axes labels and title
# unit = 'mmDay$^{-1}$'
# plt.xlabel('P-E anomaly ({unit})'.format(unit=unit))
# plt.xlim(-2.5,2.5)


# #PRECTMX plot
# ax = fig.add_subplot(414)

# var = 'PRECTMX'

# fig1_land_pop(var)
    
# # set axes labels and title
# unit = 'mmDay$^{-1}$'
# plt.xlabel('Pmax anomaly ({unit})'.format(unit=unit))
# plt.xlim(-160,160)


# """
# Figure finalizing
# """

# plt.subplots_adjust(top=0.98, bottom=0.1, left=0.10, right=0.95, hspace=0.8,
#                     wspace=0.35)

# plt.savefig(out_dir+'fig1_orig_full-glens.png', format='png', dpi=480)
# plt.savefig(out_dir+'fig1_orig_full-glens.eps', format='eps', dpi=480)

# plt.show()
