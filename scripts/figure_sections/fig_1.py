
#customize ticks
import matplotlib.ticker as ticker

"""
Figure settings
"""

out_dir = '/n/home03/pjirvine/projects/GLENS_fraction_better_off/figures/'

weight = 'land_noice_area'
out_filename = 'fig1'

# # For population-weighted version
# weight = 'pop'
# out_filename = 'fig1_pop'

quantiles = [0.01,0.05,0.25,0.5,0.75,0.95,0.99]

"""
CASES:
'Baseline'     - RCP8.5 @ 2010-2029
'RCP8.5'       - RCP8.5 @ 2075-2094
'Full-GLENS'   - GLENS  @ 2075-2094
'Half-GLENS'   - Scaled Half-GLENS  @ 2075-2094
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

"""
##################################
#
FIGURE 1 GLENS
#
##################################
"""

fig = plt.figure(figsize=cm2inch(8.5,14))

plt.rcParams.update({'font.size': 8})

def fig1_land(ax, var, mask):
    # get and then plot data
    RCP85_land = plot_data(var, 'RCP8.5', 'Baseline', weight)
    HALF_GLENS_land = plot_data(var, 'Half-GLENS', 'Baseline', weight)
    FULL_GLENS_land = plot_data(var, 'Full-GLENS', 'Baseline', weight)
    
    boxplot_3(ax, RCP85_land, HALF_GLENS_land, FULL_GLENS_land)

"""
TREFHT plot
"""
ax1 = fig.add_subplot(411)
ax=ax1

# plot data!
var = 'TREFHT'
xlims=[-2,12]
fig1_land(ax, var, weight)
    
# set axes labels and title
unit = '$^\circ$C'
plt.xlabel('T anomaly ({unit})'.format(unit=unit))
plt.xlim(xlims[0],xlims[1])

plt.text(1.3*xlims[0], 1.15, "a", clip_on=False, va="baseline", ha="left", fontsize=10, fontweight='bold')

"""
TREFHTMX plot
"""
ax2 = fig.add_subplot(412)
ax=ax2

# plot data!
var = 'TREFHTMX'
xlims=[-5,20]
fig1_land(ax, var, weight)
    
# set axes labels and title
unit = '$^\circ$C'
plt.xlabel('Tmax anomaly ({unit})'.format(unit=unit))
plt.xlim(xlims[0],xlims[1])

plt.text(1.3*xlims[0], 1.15, "b", clip_on=False, va="baseline", ha="left", fontsize=10, fontweight='bold')

"""
P-E plot
"""
ax3 = fig.add_subplot(413)
ax=ax3

# plot data!
var = 'P-E'
xlims = [-1.5,1.5]
fig1_land(ax, var, weight)
    
# set axes labels and title
unit = 'mmDay$^{-1}$'
plt.xlabel('P-E anomaly ({unit})'.format(unit=unit))
plt.xlim(xlims[0],xlims[1])

plt.text(1.15*xlims[0], 1.15, "c", clip_on=False, va="baseline", ha="left", fontsize=10, fontweight='bold')

"""
PRECTMX plot
"""
ax4 = fig.add_subplot(414)
ax=ax4

# plot data!
var = 'PRECTMX'
xlims = [-120,120]
fig1_land(ax, var, weight)
    
# set axes labels and title
unit = 'mmDay$^{-1}$'
plt.xlabel('Pmax anomaly ({unit})'.format(unit=unit))
plt.xlim(xlims[0],xlims[1])

plt.text(1.15*xlims[0], 1.15, "d", clip_on=False, va="baseline", ha="left", fontsize=10, fontweight='bold')

"""
Plot legend
"""
# use empty plots
plt.plot(0,0, color=blue, label='Full-GLENS')
plt.plot(0,0, color=purple, label='Half-GLENS')
plt.plot(0,0, color=red, label='RCP8.5')

plt.legend(frameon=False, loc=3, bbox_to_anchor=(-0.01, -0.08))

"""
Figure finalizing
"""

ax3.get_xaxis().set_ticks([-1.5,-1,-0.5,0,0.5,1.0,1.5])
ax4.get_xaxis().set_ticks([-120,-80,-40,0,40,80,120])
# ax3.get_xaxis().set_ticks([-2.5,-2,-1.5,-1,-0.5,0,0.5,1.0,1.5,2.0,2.5])
# ax4.get_xaxis().set_ticks([-160,-120,-80,-40,0,40,80,120,160])

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.10, right=0.95, hspace=0.8,
                    wspace=0.35)

plt.savefig(out_dir+out_filename+'.png', format='png', dpi=480)
plt.savefig(out_dir+out_filename+'.eps', format='eps', dpi=480)

plt.show()
