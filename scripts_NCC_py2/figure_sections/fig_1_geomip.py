"""
GeoMIP version of figure 1
"""

def plot_data_geomip(var, model, quantiles, weight_name='land_noice_area'):
    # use GFDL variable names
    
    SRM_anom = all_data[model][var]['frac_srm_CO2_effect'].flatten()
    CO2_anom = all_data[model][var]['CO2_effect'].flatten()
    
    weight = all_data[model][var][weight_name].flatten()
#     print get_weight_geomip(model, mask_type=weight_name)
    CO2_land = weighted_quantile(CO2_anom, quantiles, sample_weight=weight)
    SRM_land = weighted_quantile(SRM_anom, quantiles, sample_weight=weight)
    
    return (CO2_land, SRM_land)

def plot_data_geomip_100(var, model, quantiles, weight_name='land_noice_area'):
    # use GFDL variable names
    
    SRM_anom = all_data[model][var]['srm_CO2_effect'].flatten()
    CO2_anom = all_data[model][var]['CO2_effect'].flatten()
    
    weight = all_data[model][var][weight_name].flatten()
#     print get_weight_geomip(model, mask_type=weight_name)
    CO2_land = weighted_quantile(CO2_anom, quantiles, sample_weight=weight)
    SRM_land = weighted_quantile(SRM_anom, quantiles, sample_weight=weight)
    
    return (CO2_land, SRM_land)

def paired_boxplots(axis,y_centre,y_delta,thick,data_1,data_2,colors):
    
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
    
    # plot the shapes:
    box_rectangles(axis, data_1, y_centre + y_delta, thick, colors[0])
    box_rectangles(axis, data_2, y_centre - y_delta, thick, colors[1])

"""
FIgure setup
"""
    
quantiles = [0.01,0.05,0.25,0.5,0.75,0.95,0.99]

fig = plt.figure(figsize=cm2inch(18,28))

model_list = ['NorESM1-M','MPI-ESM-LR','MIROC-ESM','IPSL-CM5A-LR','HadGEM2-ES','HadCM3','GISS-E2-R','CSIRO-Mk3L-1-2','CESM-CAM5.1-FV','CCSM4','CanESM2','BNU-ESM']
model_x_list = ['NorESM1-M','MPI-ESM-LR','IPSL-CM5A-LR','HadGEM2-ES','GISS-E2-R','CCSM4','CanESM2','BNU-ESM']


"""
tas GeoMIP
"""

ax = fig.add_subplot(221)

var = 'tas_Amon'
nmodel = len(model_list) # use normal model list

ax.set_ylim(0,1)
ax.set_xlim(0,15) #var specific
unit = '$^\circ$C'
plt.xlabel('T anomaly ({unit})'.format(unit=unit))

# plot settings
vert_buffer, delta, width = 0.02, 0.015, 0.025 # type of var specific
vert_spacing = (1.0 - 2*vert_buffer) / nmodel
vert_loc_list = [vert_buffer + vert_spacing * 0.5 + (X) * vert_spacing for X in range(nmodel)]

for IDX in range(nmodel):
    
    model = model_list[IDX]
    CO2_land, SRM_land = plot_data_geomip(var, model, quantiles)
    paired_boxplots(ax, vert_loc_list[IDX], delta, width, SRM_land, CO2_land, [blue,red])

plt.yticks(vert_loc_list,model_list)

"""
P-E GeoMIP
"""

ax = fig.add_subplot(223)

var = 'p-e_Amon'
nmodel = len(model_list) # use normal model list

ax.set_ylim(0,1)
ax.set_xlim(-4,4) #var specific
unit = 'mmDay$^{-1}$'
plt.xlabel('PE anomaly ({unit})'.format(unit=unit))

# plot settings
vert_buffer, delta, width = 0.02, 0.015, 0.025 # type of var specific
vert_spacing = (1.0 - 2*vert_buffer) / nmodel
vert_loc_list = [vert_buffer + vert_spacing * 0.5 + (X) * vert_spacing for X in range(nmodel)]

for IDX in range(nmodel):
    
    model = model_list[IDX]
    CO2_land, SRM_land = plot_data_geomip(var, model, quantiles)
    paired_boxplots(ax, vert_loc_list[IDX], delta, width, SRM_land, CO2_land, [blue,red])

plt.plot([0,0],[0,1], color='k', zorder=0)
    
plt.yticks(vert_loc_list,model_list)

"""
tas max GeoMIP
"""

ax = fig.add_subplot(222)

var = 'txxETCCDI_yr'
nmodel = len(model_x_list) # use normal model list

ax.set_ylim(0,1)
ax.set_xlim(0,20) #var specific
unit = '$^\circ$C'
plt.xlabel('Tx anomaly ({unit})'.format(unit=unit))

# plot settings
vert_buffer, delta, width = 0.02, 0.015, 0.025 # type of var specific
vert_spacing = (1.0 - 2*vert_buffer) / nmodel
vert_loc_list = [vert_buffer + vert_spacing * 0.5 + (X) * vert_spacing for X in range(nmodel)]

for IDX in range(nmodel):
    
    model = model_x_list[IDX]
    CO2_land, SRM_land = plot_data_geomip(var, model, quantiles)
    paired_boxplots(ax, vert_loc_list[IDX], delta, width, SRM_land, CO2_land, [blue,red])

plt.yticks(vert_loc_list,model_x_list)

"""
precip max GeoMIP
"""

ax = fig.add_subplot(224)

var = 'rx5dayETCCDI_yr'
nmodel = len(model_x_list) # use normal model list

ax.set_ylim(0,1)
ax.set_xlim(-100,200) #var specific
unit = 'mmDay$^{-1}$'
plt.xlabel('Px anomaly ({unit})'.format(unit=unit))

# plot settings
vert_buffer, delta, width = 0.02, 0.015, 0.025 # type of var specific
vert_spacing = (1.0 - 2*vert_buffer) / nmodel
vert_loc_list = [vert_buffer + vert_spacing * 0.5 + (X) * vert_spacing for X in range(nmodel)]

for IDX in range(nmodel):
    
    model = model_x_list[IDX]
    CO2_land, SRM_land = plot_data_geomip(var, model, quantiles)
    paired_boxplots(ax, vert_loc_list[IDX], delta, width, SRM_land, CO2_land, [blue,red])

plt.yticks(vert_loc_list,model_x_list)

plt.plot([0,0],[0,1], color='k', zorder=0)

"""
Tidy up figure
"""

plt.subplots_adjust(top=0.98, bottom=0.05, left=0.20, right=0.95, hspace=0.2,
                    wspace=0.55)

plt.savefig(out_dir+'fig1_geomip.png', format='png', dpi=480)
plt.savefig(out_dir+'fig1_geomip.eps', format='eps', dpi=480)

plt.show()

"""
###
###
End of 50% GeoMIP figure
###
###
"""




"""
G1 version of fig1 GeoMIP
"""

fig = plt.figure(figsize=cm2inch(18,28))

"""
tas GeoMIP
"""

ax = fig.add_subplot(221)

var = 'tas_Amon'
nmodel = len(model_list) # use normal model list

ax.set_ylim(0,1)
ax.set_xlim(-5,15) #var specific
unit = '$^\circ$C'
plt.xlabel('T anomaly ({unit})'.format(unit=unit))

# plot settings
vert_buffer, delta, width = 0.02, 0.015, 0.025 # type of var specific
vert_spacing = (1.0 - 2*vert_buffer) / nmodel
vert_loc_list = [vert_buffer + vert_spacing * 0.5 + (X) * vert_spacing for X in range(nmodel)]

plt.plot([0,0],[0,1], color='k', zorder=0)

for IDX in range(nmodel):
    
    model = model_list[IDX]
    CO2_land, SRM_land = plot_data_geomip_100(var, model, quantiles)
    paired_boxplots(ax, vert_loc_list[IDX], delta, width, SRM_land, CO2_land, [blue,red])

plt.yticks(vert_loc_list,model_list)

"""
P-E GeoMIP
"""

ax = fig.add_subplot(223)

var = 'p-e_Amon'
nmodel = len(model_list) # use normal model list

ax.set_ylim(0,1)
ax.set_xlim(-4,4) #var specific
unit = 'mmDay$^{-1}$'
plt.xlabel('PE anomaly ({unit})'.format(unit=unit))

# plot settings
vert_buffer, delta, width = 0.02, 0.015, 0.025 # type of var specific
vert_spacing = (1.0 - 2*vert_buffer) / nmodel
vert_loc_list = [vert_buffer + vert_spacing * 0.5 + (X) * vert_spacing for X in range(nmodel)]

for IDX in range(nmodel):
    
    model = model_list[IDX]
    CO2_land, SRM_land = plot_data_geomip_100(var, model, quantiles)
    paired_boxplots(ax, vert_loc_list[IDX], delta, width, SRM_land, CO2_land, [blue,red])

plt.plot([0,0],[0,1], color='k', zorder=0)
    
plt.yticks(vert_loc_list,model_list)

"""
tas max GeoMIP
"""

ax = fig.add_subplot(222)

var = 'txxETCCDI_yr'
nmodel = len(model_x_list) # use normal model list

ax.set_ylim(0,1)
ax.set_xlim(-5,20) #var specific
unit = '$^\circ$C'
plt.xlabel('Tx anomaly ({unit})'.format(unit=unit))

plt.plot([0,0],[0,1], color='k', zorder=0)

# plot settings
vert_buffer, delta, width = 0.02, 0.015, 0.025 # type of var specific
vert_spacing = (1.0 - 2*vert_buffer) / nmodel
vert_loc_list = [vert_buffer + vert_spacing * 0.5 + (X) * vert_spacing for X in range(nmodel)]

for IDX in range(nmodel):
    
    model = model_x_list[IDX]
    CO2_land, SRM_land = plot_data_geomip_100(var, model, quantiles)
    paired_boxplots(ax, vert_loc_list[IDX], delta, width, SRM_land, CO2_land, [blue,red])

plt.yticks(vert_loc_list,model_x_list)

"""
precip max GeoMIP
"""

ax = fig.add_subplot(224)

var = 'rx5dayETCCDI_yr'
nmodel = len(model_x_list) # use normal model list

ax.set_ylim(0,1)
ax.set_xlim(-100,200) #var specific
unit = 'mmDay$^{-1}$'
plt.xlabel('Px anomaly ({unit})'.format(unit=unit))

# plot settings
vert_buffer, delta, width = 0.02, 0.015, 0.025 # type of var specific
vert_spacing = (1.0 - 2*vert_buffer) / nmodel
vert_loc_list = [vert_buffer + vert_spacing * 0.5 + (X) * vert_spacing for X in range(nmodel)]

for IDX in range(nmodel):
    
    model = model_x_list[IDX]
    CO2_land, SRM_land = plot_data_geomip_100(var, model, quantiles)
    paired_boxplots(ax, vert_loc_list[IDX], delta, width, SRM_land, CO2_land, [blue,red])

plt.yticks(vert_loc_list,model_x_list)

plt.plot([0,0],[0,1], color='k', zorder=0)

"""
Tidy up figure
"""

plt.subplots_adjust(top=0.98, bottom=0.05, left=0.20, right=0.95, hspace=0.2,
                    wspace=0.55)

plt.savefig(out_dir+'fig1_geomip_100.png', format='png', dpi=480)
plt.savefig(out_dir+'fig1_geomip_100.eps', format='eps', dpi=480)

plt.show()