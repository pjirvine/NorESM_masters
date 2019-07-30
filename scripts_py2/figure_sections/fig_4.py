
"""
Figure to plot RMS, etc. as a function of solar constant reduction
"""

out_dir = '/n/home03/pjirvine/projects/fraction_better_off/figures/fig_4/'

# load up GeoMIP regional data
table_dir = '/n/home03/pjirvine/projects/fraction_better_off/tables/'
frac_pd = pd.DataFrame.from_csv(table_dir + 'results_by_frac_geo.csv')
frac_dict = frac_pd.to_dict()

vars_hiflor = ['tas','tasmax','precip','precip5max','pe']
vars_no_p = ['tas','tasmax','precip5max','pe']

metrics = ['global','RMS','RMS_std','mod','exa']

x = frac_dict['pe_RMS'].keys()

"""
Begin Plots
"""

fig = plt.figure(figsize=cm2inch(8.5,18))
plt.rcParams.update({'font.size': 8})

var_cols = {'tas':'red',
            'tasmax':'purple',
            'precip5max':'blue',
            'pe':'green'}

var_labels = {'tas':'T',
               'tasmax':'Tx',
               'precip5max':'Px',
               'pe':'PE'}

"""
Global-mean plot
"""

ax1 = fig.add_subplot(411)

# generate control global-mean precip
control_precip = gfdl_data['ctrl','precip','mean'].flatten()
global_weight = gfdl_masks['area'].flatten()
control_global_precip = np.sum(control_precip * global_weight)

global_temp = frac_pd['tas_global']
global_precip_pc = 100.0 * (frac_pd['precip_global'] / control_global_precip)

plt.plot(global_temp, color = var_cols['tas'], label= var_labels['tas'])
plt.plot(global_precip_pc, color = 'k', label = 'P')

plt.xlim(0,2)
plt.ylim(-3,4)

plt.axhline(0.,color='gray',zorder=0)
plt.axvline(1.,color='gray',zorder=0)

plt.title('Global-mean Anomaly')
plt.ylabel('Anomaly ($^\circ$C and %)')

plt.legend(frameon=False)

"""
RMS plot
"""

ax2 = fig.add_subplot(412)

for var in vars_no_p:
    
    RMS = frac_pd[var+'_RMS'] / frac_pd[var+'_RMS'][0]
    plt.plot(RMS, color = var_cols[var], label= var_labels[var])

plt.ylim(0,1)
plt.xlim(0,2)

plt.axvline(1.,color='gray',zorder=0)

plt.title('RMS Anomaly from Control')
plt.ylabel('Normalized Units')


plt.legend(frameon=False)

"""
Fraction moderated plot
"""

ax3 = fig.add_subplot(413)

for var in vars_no_p:
    
    mod = 100. * frac_pd[var+'_mod']
    plt.plot(mod, color = var_cols[var], label= var_labels[var])

plt.axvline(1.,color='gray',zorder=0)
    
plt.ylim(0,100)
plt.xlim(0,2)

plt.title('Fraction Moderated')
plt.ylabel('Fraction (%)')

# plt.legend(frameon=False)

"""
Fraction exacerbated plot
"""

ax4 = fig.add_subplot(414)

for var in vars_no_p:
    
    exa = 100. * frac_pd[var+'_exa']
    plt.plot(exa, color = var_cols[var], label= var_labels[var])

plt.text(0.88,7,'half-SG',rotation=90)
plt.axvline(1.,color='gray',zorder=0)
    
plt.ylim(0,10)
plt.xlim(0,2)

plt.title('Fraction Exacerbated')
plt.ylabel('Fraction (%)')
plt.xlabel('Solar Constant Reduction (%)')

# plt.legend(frameon=False)

"""
Tidy up figure
"""

ax1.get_xaxis().set_ticklabels([])
ax2.get_xaxis().set_ticklabels([])
ax3.get_xaxis().set_ticklabels([])

ax1.tick_params(axis='y', right=True)
ax2.tick_params(axis='y', right=True)
ax3.tick_params(axis='y', right=True)
ax4.tick_params(axis='y', right=True)

ax2.get_yaxis().set_ticks([0.0,0.2,0.4,0.6,0.8,1.0])
ax4.get_yaxis().set_ticks([0, 2, 4, 6, 8, 10])

fig.subplots_adjust(left=0.18, right=0.95, hspace=0.3)

plt.savefig(out_dir+'fig_4.png', format='png', dpi=480)
plt.savefig(out_dir+'fig_4.eps', format='eps', dpi=480)

plt.show()

# 