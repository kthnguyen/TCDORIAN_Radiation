import numpy as np 
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import pangeo

WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
IMDIR = WORKPLACE + r"\CCMP_Wind"
FIGDIR = IMDIR + r"\Figures"

Btracks = xr.open_dataset(WORKPLACE+r"\2013204N11340.ibtracs.v03r10.nc")

#Extract variables into arrays
Btime = Btracks['time'].values
Byear = pd.to_datetime(Btime).year
Bmonth = pd.to_datetime(Btime).month
Bday = pd.to_datetime(Btime).day
Bhour = pd.to_datetime(Btime).hour
Blat = Btracks['lat_for_mapping'].values
Blon = Btracks['lon_for_mapping'].values

#Interpolate best track lat long to 0.5-hour intervals
df = pd.DataFrame({'time':Btime,'lat':Blat,'lon':Blon})
df = df.set_index('time')
df_reindexed = df.reindex(pd.date_range(start=Btime[0],end=Btime[len(Btime)-1],freq='6H'))
df_reindexed = df_reindexed.interpolate(method='time')
df_reindexed.index.name = 'time'
df_reindexed.reset_index(inplace = True)

#%%
def convert_coords(coord_array, option):
    if option == "to180":
        for i in range(0,coord_array.size):
            if coord_array[i] >180:
                coord_array[i] = coord_array[i]-360
    if option == "to360":
        for i in range(0,coord_array.size):
            if coord_array[i] <0:
                coord_array[i] = coord_array[i]+360
#%%
#pick an instance
for i in range(0,df_reindexed.shape[0]):
#for i in range(0,1):
    step = df_reindexed.iloc[i]
    t_year = pd.to_datetime(step.time).year
    t_month = pd.to_datetime(step.time).month
    t_day = pd.to_datetime(step.time).day
    t_hour = pd.to_datetime(step.time).hour
    t_minute = pd.to_datetime(step.time).minute
    t_lat = step.lat
    t_lon = step.lon
    
    #getname to match with timing the corresponding IR image name
    str_t_year = str(t_year)
    
    if t_month < 10:
        str_t_month = "0" + str(t_month)
    else:
        str_t_month = str(t_month)

    if t_day < 10:
        str_t_day = "0" + str(t_day)
    else:
        str_t_day = str(t_day)      
    
    if t_hour < 10:
        str_t_hour = "0" + str(t_hour)
    else:
        str_t_hour = str(t_hour) 
        
    if t_minute < 10:
        str_t_minute = "0" + str(t_minute)
    else:
        str_t_minute = str(t_minute)
        
    t_time_no_hour = str_t_year + str_t_month + str_t_day
    t_time = str_t_year + str_t_month + str_t_day + str_t_hour
    t_time_withmin = t_time + str_t_minute
    #get path for the corresponding IR img and open it
    t_IR_path = IMDIR + "\CCMP_Wind_Analysis_" + t_time_no_hour + "_V02.0_L3.0_RSS.nc"
    t_IRimg = xr.open_dataset(t_IR_path)
#    t_IRimg = xr.open_dataarray(t_IR_path, drop_variables = ['vwnd', 'nobs'])
    
    #10deg distance from center of TC
    lat_bounds = np.float32([t_lat-20, t_lat+20])
    lon_bounds = np.float32([t_lon-20, t_lon+20])
    
    t_IRimg_lat = t_IRimg.coords['latitude'].values
    t_IRimg_lon = t_IRimg.coords['longitude'].values
    
    convert_coords(t_IRimg_lon,"to180")
        
    t_IRimg_time = t_IRimg.coords['time'].values
    
    t_IRimg_lat = np.squeeze(t_IRimg_lat)
    t_IRimg_lon = np.squeeze(t_IRimg_lon)
    
    #get min index that within the bounds
    min_lat_inds = min([i for i,x in enumerate(t_IRimg_lat) if (x>lat_bounds[0] and x<lat_bounds[1])])
    min_lon_inds = min([i for i,x in enumerate(t_IRimg_lon) if (x>lon_bounds[0] and x<lon_bounds[1])])
    
    #take 1101 elements counting from the mins
    lat_inds = t_IRimg_lat[min_lat_inds:min_lat_inds+159]
    lon_inds = t_IRimg_lon[min_lon_inds:min_lon_inds+159]
    
    convert_coords(lon_inds,"to360")
    #remap the IR image
    if t_hour == 0:
        t_IRimg_remap = t_IRimg.sel(time=t_IRimg_time[0], longitude = lon_inds, latitude = lat_inds, method='nearest')
        t_IRimg_remap = t_IRimg_remap.expand_dims('time')
        t_IRimg_remap.to_netcdf(path = IMDIR + r"\Remap\TCDorian_wind_" + t_time_withmin + ".nc", mode = 'w', format = "NETCDF4" )
        print(t_time_withmin + " done")
    elif t_hour == 6:
        t_IRimg_remap = t_IRimg.sel(time=t_IRimg_time[1], longitude = lon_inds, latitude = lat_inds, method='nearest')
        t_IRimg_remap = t_IRimg_remap.expand_dims('time')
        t_IRimg_remap.to_netcdf(path = IMDIR + r"\Remap\TCDorian_wind_" + t_time_withmin + ".nc", mode = 'w', format = "NETCDF4" )
        print(t_time_withmin + " done")
    elif t_hour== 12: 
        t_IRimg_remap = t_IRimg.sel(time=t_IRimg_time[2], longitude = lon_inds, latitude = lat_inds, method='nearest')
        t_IRimg_remap = t_IRimg_remap.expand_dims('time')
        t_IRimg_remap.to_netcdf(path = IMDIR + r"\Remap\TCDorian_wind_" + t_time_withmin + ".nc", mode = 'w', format = "NETCDF4" )
        print(t_time_withmin + " done")
    elif t_hour == 18:
        t_IRimg_remap = t_IRimg.sel(time=t_IRimg_time[3], longitude = lon_inds, latitude = lat_inds, method='nearest')
        t_IRimg_remap = t_IRimg_remap.expand_dims('time')
        t_IRimg_remap.to_netcdf(path = IMDIR + r"\Remap\TCDorian_wind_" + t_time_withmin + ".nc", mode = 'w', format = "NETCDF4" )
        print(t_time_withmin + " done")
        # interpolate NaN values
        #output
        

   #%%    
    # Save image
    uwind_val = t_IRimg_remap.uwnd.values
    vwind_val = t_IRimg_remap.vwnd.values
    temp_IRimg_lon = t_IRimg_remap.coords['longitude'].values
    temp_IRimg_lat = t_IRimg_remap.coords['latitude'].values
    temp_IRimg_time = t_IRimg_remap.coords['time'].values
    temp_IRimg_lat = np.round(np.squeeze(temp_IRimg_lat),2)
    temp_IRimg_lon = np.round(np.squeeze(temp_IRimg_lon),2)
    
    t_lat = step.lat
    t_lon = step.lon
    fig = plt.figure()
    SMALL_SIZE = 6
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    
    plt.subplot(121)
    im = plt.imshow(uwind_val, extent = (temp_IRimg_lon.min(),temp_IRimg_lon.max(), temp_IRimg_lat.min(),temp_IRimg_lat.max()), cmap='coolwarm',origin='lower',animated=True)
    cb = fig.colorbar(im, orientation='horizontal',)
    cb.set_label(r'SS+10m Wind speed (m/s)')
    
#    plt.plot(t_lon+360,t_lat,'go', markersize = 2)  
    ax = plt.gca()
    ax.set_xlabel('Longtitude')
    ax.set_ylabel('Latitude')
    ax.set_title('TC Dorian ' + t_time + "00" + r" Uwind (eastward)")
    
    #plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    #plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    #plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    
    plt.subplot(122)
    im = plt.imshow(vwind_val, extent = (temp_IRimg_lon.min(),temp_IRimg_lon.max(), temp_IRimg_lat.min(),temp_IRimg_lat.max()), cmap='coolwarm',origin='lower',animated=True)
    cb = fig.colorbar(im, orientation='horizontal')
    cb.set_label(r'SS+10m Wind speed (m/s)')
    
#    plt.plot(t_lon+360,t_lat,'go', markersize = 2)  
    ax = plt.gca()
    ax.set_xlabel('Longtitude')
    ax.set_ylabel('Latitude')
    ax.set_title('TC Dorian ' + t_time + "00" + r" Vwind (northward)")
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    
    fig.set_size_inches(7, 3)
    
#    fig.savefig(FIGDIR + r"\wind_" + t_time +".png",dpi=600)
#    plt.close()
    plt.show()
#IRdataset = xr.open_dataset(r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina\IRimages_remap_interpolate_masked"+"\\" + t_time_withmin +".nc")        
#
#IRdataset_exp = IRdataset.expand_dims('time')
# =============================================================================
#%% #temporary print
uwind_val = t_IRimg_remap.uwnd.values
vwind_val = t_IRimg_remap.vwnd.values
temp_IRimg_lon = t_IRimg_remap.coords['longitude'].values
temp_IRimg_lat = t_IRimg_remap.coords['latitude'].values
temp_IRimg_time = t_IRimg_remap.coords['time'].values
temp_IRimg_lat = np.round(np.squeeze(temp_IRimg_lat),2)
temp_IRimg_lon = np.round(np.squeeze(temp_IRimg_lon),2)

t_lat = step.lat
t_lon = step.lon
fig = plt.figure()
SMALL_SIZE = 6
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.subplot(121)
im = plt.imshow(uwind_val, extent = (temp_IRimg_lon.min(),temp_IRimg_lon.max(), temp_IRimg_lat.min(),temp_IRimg_lat.max()), cmap='coolwarm',origin='lower',animated=True)
cb = fig.colorbar(im, orientation='horizontal',)
cb.set_label(r'SS+10m Wind speed (m/s)')

plt.plot(t_lon+360,t_lat,'go', markersize = 2)  
ax = plt.gca()
ax.set_xlabel('Longtitude')
ax.set_ylabel('Latitude')
ax.set_title('TC Dorian ' + t_time + "00" + r" Uwind (eastward)")

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

plt.subplot(122)
im = plt.imshow(vwind_val, extent = (temp_IRimg_lon.min(),temp_IRimg_lon.max(), temp_IRimg_lat.min(),temp_IRimg_lat.max()), cmap='coolwarm',origin='lower',animated=True)
cb = fig.colorbar(im, orientation='horizontal')
cb.set_label(r'SS+10m Wind speed (m/s)')

plt.plot(t_lon+360,t_lat,'go', markersize = 2)  
ax = plt.gca()
ax.set_xlabel('Longtitude')
ax.set_ylabel('Latitude')
ax.set_title('TC Dorian ' + t_time + "00" + r" Vwind (northward)")

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

fig.set_size_inches(7, 3)

fig.savefig(FIGDIR + r"\wind_" + t_time +".png",dpi=600)
plt.close()
#plt.show()
# =============================================================================
#test_lat_inds = min([i for i,x in enumerate(t_IRimg_lat) if (x>lat_bounds[0] and x<lat_bounds[1])])
