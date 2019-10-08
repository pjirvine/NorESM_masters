
import regionmask
import cartopy.crs as ccrs

def mini_panels(axis, anom_1, anom_2, ttest_anom, x_loc, y_loc, text_1, text_2, half_width = 8, displace = None):

    if displace is not None:
        
        x_loc_orig, y_loc_orig = x_loc, y_loc
        
        x_loc = x_loc + displace[0]
        y_loc = y_loc + displace[1]
        
        axis.plot([x_loc,x_loc_orig],[y_loc,y_loc_orig],'k',linewidth=3, zorder=1)
    
    big_anom = max(abs(anom_1),abs(anom_2))
    norm_value = max(big_anom,abs(2.*ttest_anom))
    
    norm_anom_1 = anom_1 / norm_value
    norm_anom_2 = anom_2 / norm_value
    norm_ttest_anom = ttest_anom / norm_value
    
    thick = 0.3
    bar_loc = 0.6
    
    # create a rectangle
    patches = [
        # Background
        mpatches.Rectangle((x_loc - 1.05*half_width,y_loc - 1.15*half_width), 2.1*half_width, 2.6*half_width, facecolor='k', linewidth=0, zorder=2),
        # Background
        mpatches.Rectangle((x_loc - half_width,y_loc - 1.1*half_width), 2*half_width, 2.5*half_width, facecolor='white', linewidth=0, zorder=2),
        # Ttest
        mpatches.Rectangle((x_loc - half_width,y_loc - norm_ttest_anom * half_width), 2*half_width, 2.* norm_ttest_anom * half_width, facecolor='gray', linewidth=0, zorder=2),
        # Anom_1
        mpatches.Rectangle((x_loc - (bar_loc + 0.5*thick) * half_width,y_loc), thick*half_width, norm_anom_1 * half_width, facecolor='r', linewidth=0, zorder=3),
        # Anom_2
        mpatches.Rectangle((x_loc + (bar_loc - 0.5*thick) * half_width,y_loc), thick*half_width, norm_anom_2 * half_width, facecolor='b', linewidth=0, zorder=3),        
    ]
    for p in patches:
        axis.add_patch(p)
    
    #ttest_anom bar
#     axis.fill_between([x_loc - half_width,x_loc + half_width],[y_loc + norm_ttest_anom * half_width,y_loc + norm_ttest_anom * half_width],[y_loc - norm_ttest_anom * half_width,y_loc - norm_ttest_anom * half_width],'gray',linewidth=0)
    
    #zero line
    axis.plot([x_loc - half_width,x_loc + half_width],[y_loc,y_loc],'k',linewidth=1, zorder=4)

    #Between line
    axis.plot([x_loc - (bar_loc * half_width), x_loc + (bar_loc * half_width)],[y_loc + (norm_anom_1 * half_width),y_loc + (norm_anom_2 * half_width)],'k',linewidth=1)
    
    #Half-way Point
    axis.plot([x_loc],[y_loc + 0.5 * (norm_anom_1 + norm_anom_2) * half_width],color='purple', marker='.', markersize=15)
    
    #text
    axis.text(x_loc - bar_loc * half_width, y_loc + 1.05*half_width, text_1,  horizontalalignment='center', verticalalignment='bottom', fontsize=8)
    axis.text(x_loc + bar_loc * half_width, y_loc + 1.05*half_width, text_2,  horizontalalignment='center', verticalalignment='bottom', fontsize=8)
    
"""
Create SREX mask used as base for summary plot
"""

plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.figsize': (18,9)}) # Square panels (2 across a page)

ax = regionmask.defined_regions.srex.plot(add_label=False, line_kws={'zorder':1, 'linewidth':1})
plt.tight_layout()

region_centres_list = [[-136.511,   66.277],[-57.5,  67.5],[-117.5  ,   44.283],[-95.   ,  39.283],[-72.5,  37.5],[-90.25802898,  16.60289305],[-62.05181777,  -3.75447446],[-42., -10.],[-75.89741775, -30.77603057],[-54.40601015, -38.77303345],[12.27138643, 64.46654867],[20.74534161, 50.5952795 ],[15. , 37.5],[10. , 22.5],[2.5   , 1.8175],[38.495 ,  1.8175],[ 20.995 , -23.1825],[110.,  60.],[50. , 32.5],[67.5, 40. ],[87.5, 40. ],[122.5,  35. ],[78.58108108, 17.90540541],[125.,   5.],[132.5, -20. ],[145., -40.]]

def example_plot(region_centre):
    mini_panels(ax, 5., -3., 1.5, region_centre[0], region_centre[1], '+5.0','-3.0')

for region_centre in region_centres_list:
    example_plot(region_centre)
    
plt.savefig('SREX_regions.eps', format='eps', dpi=480)
plt.savefig('SREX_regions.png', format='png', dpi=480)
plt.show()
