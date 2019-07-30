
"""
Define rectangle drawing function

useful guide here:
http://matthiaseisen.com/pp/patterns/p0203/

"""

import ast

def region_data(region):
    
    """
    GeoMIP data
    """
    
    # load up GeoMIP regional data
    geo_table_dir = '/n/home03/pjirvine/projects/fraction_better_off/tables/geomip_regional_means/'
    geomip_dict = pd.read_csv(geo_table_dir + 'GeoMIP_region_better_list.csv',index_col=0).to_dict()

    pe_bw_models = ast.literal_eval(geomip_dict['p-e_Amon'][region])
    px_bw_models = ast.literal_eval(geomip_dict['rx5dayETCCDI_yr'][region])
    
    # [B_list, b_list, w_list, W_list]

    pe_bw_count = [len(X) for X in pe_bw_models]
    px_bw_count = [len(X) for X in px_bw_models]
    
    """
    GFDL data
    """
    
    in_dir = '/n/home03/pjirvine/projects/fraction_better_off/tables/'
    gfdl_dict = pd.read_csv(in_dir+'anom_SREX.csv',index_col=0, header=[0,1]).to_dict()

    # copy gfdl table data
    dict_clean = gfdl_dict

    def gfdl_color(var):
        
        bw_off = dict_clean[(var,'better_worse')]
        CO2_anom = dict_clean[(var,'CO2-ctrl')]
        SRM_anom = dict_clean[(var,'SRM-ctrl')]
        
       
        if bw_off[region] == 1: # BETTER
            return blue
        elif bw_off[region] == -1: # WORSE
            return red
        else: # not significant BUT...
            if abs(CO2_anom[region]) > abs(SRM_anom[region]):
                return l_blue # CO2 > SRM (better)
            else:
                return l_red # CO2 < SRM (worse)
           
    pe_color = gfdl_color('pe')
    px_color = gfdl_color('precip5max')
    
    return pe_color, px_color, pe_bw_count, px_bw_count
# end region data

def vert_stack(x_start, y_start, width, height, data):

    BETTER = 1. * data[0] / sum(data)
    better = 1. * data[1] / sum(data)
    worse = 1. * data[2] / sum(data)
    WORSE = 1. * data[3] / sum(data)
    
    y_XXX = y_start # start of WORSE
    y_WXX = y_start + (WORSE) * height # start of worse
    y_WwX = y_start + (WORSE + worse) * height # start of better
    y_Wwb = y_start + (WORSE + worse + better) * height # start of BETTER

    patches = [
        mpatches.Rectangle((x_start,y_start), width, height, fill=False, facecolor='w', linewidth=0), ### Background
        mpatches.Rectangle((x_start,y_XXX), width, height*WORSE, facecolor=red, linewidth=0), 
        mpatches.Rectangle((x_start,y_WXX), width, height*worse, facecolor='w', linewidth=0), # white background for worse
        mpatches.Rectangle((x_start,y_WXX), width, height*worse, facecolor=red, linewidth=0, alpha=0.15), 
        mpatches.Rectangle((x_start,y_WwX), width, height*better, facecolor='w', linewidth=0), # white background for better
        mpatches.Rectangle((x_start,y_WwX), width, height*better, facecolor=blue, linewidth=0, alpha=0.15), 
        mpatches.Rectangle((x_start,y_Wwb), width, height*BETTER, facecolor=blue, linewidth=0), 
        ]

    return patches

def text_centre(text, x_start, y_start, width, height, color='b'):
    plt.text(x_start + 0.5*width, y_start + 0.5*height, text, ha="center", va='center', family='sans-serif', size=12)
    return [mpatches.Rectangle((x_start,y_start), width, height, facecolor=color, linewidth=0)]

def double_vert_plus(x_left, y_bottom, geomip_pe, geomip_px, pe_color, px_color, label):
    
    patches_return = []
    
    width = 0.3 # width of elements
    gap = 0.05 # gap between visual elements
    height_bar = 0.65 # height of bar
    height_top = 0.15 # height of top part
    
    # coordinates for bounding frame (just wider than elements)
    x_frame_start = x_left
    y_frame_start = y_bottom
    x_frame_len = gap + width * 2
    y_frame_len = gap + height_bar + height_top
    frame_color = 'white'
    
    # add frame patch
    patches_return.append(mpatches.Rectangle((x_frame_start,y_frame_start), x_frame_len, y_frame_len, facecolor=frame_color, linewidth=0) )
    
    # coordinates for label
    x_label = x_frame_start - gap*5
    y_label = y_frame_start + gap*2
    
    plt.text(x_label, y_label, label, ha="center", va='center', family='sans-serif', size=6)

    # define start of right and upper elements
    x_left_2 = x_left + width + gap
    y_bottom_2 = y_bottom + height_bar + gap
    
    # GFDL PE                       X0    Y0    Xlen Ylen
#     patches_return.extend(text_centre('PE', x_left, y_bottom_2, width, height_top, color=blue)) with text version
    patches_return.append(mpatches.Rectangle((x_left,y_bottom_2), width, height_top, facecolor=pe_color, linewidth=0) )
    # GeoMIP PE
    patches_return.extend(vert_stack(x_left, y_bottom, width, height_bar, geomip_pe))

    # GFDL PX
#     patches_return.extend(text_centre('PX', x_left_2, y_bottom_2, width, height_top, color=blue)) with text version
    patches_return.append(mpatches.Rectangle((x_left_2,y_bottom_2), width, height_top, facecolor=px_color, linewidth=0) )

    # GeoMIP Px
    patches_return.extend(vert_stack(x_left_2, y_bottom, width, height_bar, geomip_px))
    
    return patches_return

"""
Start test figures
"""

# example data:

gfdl_pe = [1,0,0,0]
gfdl_px = [1,0,0,0]

geomip_pe = [5,4,1,2]
geomip_px = [3,4,1,0]

# start figure

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')

patches = []

grid = np.mgrid[0.5:6.5:6j, 0.5:5.5:6j].reshape(2, -1).T

region_num_dict = {1: 'ALA', 2: 'AMZ', 3: 'CAM', 4: 'CAS', 5: 'CEU', 
                   6: 'CGI', 7: 'CNA', 8: 'EAF', 9: 'EAS', 10: 'ENA', 
                   11: 'MED', 12: 'NAS', 13: 'NAU', 14: 'NEB', 15: 'NEU', 
                   16: 'SAF', 17: 'SAH', 18: 'SAS', 19: 'SAU', 20: 'SEA', 
                   21: 'SSA', 22: 'TIB', 23: 'WAF', 24: 'WAS', 25: 'WNA', 
                   26: 'WSA'}

nums = np.arange(1,27)

for num in nums:

    region = region_num_dict[num] #region name
    
    # load plot data for this region
    pe_color, px_color, pe_bw_count, px_bw_count = region_data(region)
    
    patches_temp = double_vert_plus(grid[num][0], grid[num][1], pe_bw_count, px_bw_count, pe_color, px_color, region)
    patches.extend(patches_temp)

for p in patches:
    ax1.add_patch(p)

plt.axis([0,7,0,7])
    
plt.savefig(out_dir+'fig_3_tabs.png', format='png', dpi=480)
plt.savefig(out_dir+'fig_3_tabs.eps', format='eps', dpi=480) 

plt.show()