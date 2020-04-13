
# Function to calculate global mean of given file / variable
def global_mean_data(filename, variable, weight):
    data = Dataset(filename)
    time_mean = np.mean(data.variables[variable][:],axis=0)
    global_mean = np.sum(time_mean * weight)
    return global_mean

raw_daily_data_dir = "/n/home03/pjirvine/keithfs1_pji/GLENS/raw_daily_data/"
raw_monthly_data_dir = "/n/home03/pjirvine/keithfs1_pji/GLENS/raw_monthly_data/"

daily_nc_format = "{exp}.001.cam.h3.{var}.20100101-20191230.nc"
monthly_nc_format = "{exp}.001.cam.h0.{var}.201001-201912.nc"

daily_vars = ['PRECT','LHFLX']
monthly_vars = ['PRECC','PRECL','LHFLX','QFLX']
exps = ['control','feedback']

####

exp = 'control'
weight = np.transpose(all_masks['area'])
daily_prect = global_mean_data(raw_daily_data_dir + daily_nc_format.format(exp=exp, var='PRECT'), 'PRECT', weight)

monthly_precl = global_mean_data(raw_monthly_data_dir + monthly_nc_format.format(exp=exp, var='PRECL'), 'PRECL', weight)
monthly_precc = global_mean_data(raw_monthly_data_dir + monthly_nc_format.format(exp=exp, var='PRECC'), 'PRECC', weight)

monthly_LHFLX = global_mean_data(raw_monthly_data_dir + monthly_nc_format.format(exp=exp, var='LHFLX'), 'LHFLX', weight)
monthly_QFLX = global_mean_data(raw_monthly_data_dir + monthly_nc_format.format(exp=exp, var='QFLX'), 'QFLX', weight)

print('daily:', daily_prect, 'monthly:',monthly_precl + monthly_precc)
print('monthly qflx', monthly_QFLX, 'monthly lhflx', monthly_LHFLX)
#####

# control_daily_PRECT_nc = "control.001.cam.h3.PRECT.20100101-20191230.nc"
# feedback_daily_PRECT_nc = "feedback.001.cam.h3.PRECT.20200101-20291231.nc"

# control_daily_LHFLX_nc = "control.001.cam.h3.LHFLX.20100101-20191230.nc"
# feedback_daily_LHFLX_nc = "feedback.001.cam.h3.LHFLX.20200101-20291231.nc"

# feedback_monthly_PRECC_nc = "feedback.001.cam.h0.PRECC.202001-202912.nc"

print("LH in WM-2:",monthly_LHFLX)
print("L = 2.5 MJ kg-1 (latent heat of evaporation of water)")
print("PRECT in Ms-1:",daily_prect)
print("PRECT in Kgs-1:",daily_prect*1000.)
print("LH in WM-2 / PRECT in Kgs-1 = L:",monthly_LHFLX / (daily_prect*1000.))
print("L = 2.5 MJ kg-1 (latent heat of evaporation of water)")
