import numpy as np 
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
IRDIR = WORKPLACE + r"\IR_images"

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
df_reindexed = df.reindex(pd.date_range(start=Btime[0],end=Btime[len(Btime)-1],freq='0.5H'))
df_reindexed = df_reindexed.interpolate(method='time')
df_reindexed.index.name = 'time'
df_reindexed.reset_index(inplace = True)

#pick an instance
#for i in range(0,df_reindexed.shape[0]):
for i in range(0,1):
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
    
    t_time = str_t_year + str_t_month + str_t_day + str_t_hour
    t_time_withmin = t_time + str_t_minute
    #get path for the corresponding IR img and open it
    t_IR_path = IRDIR + "\merg_" + t_time + "_4km-pixel.nc4"
    t_IRimg = xr.open_dataset(t_IR_path)
    
    #(lat(-10,50), lon(-100,0)
    lat_bounds = np.float32([-10, 50])
    lon_bounds = np.float32([-100, 0])

    t_IRimg_lat = t_IRimg.coords['lat'].values
    t_IRimg_lon = t_IRimg.coords['lon'].values
    t_IRimg_time = t_IRimg.coords['time'].values
    
    t_IRimg_lat = np.squeeze(t_IRimg_lat)
    t_IRimg_lon = np.squeeze(t_IRimg_lon)
    
    #get min index that within the bounds
    min_lat_inds = min([i for i,x in enumerate(t_IRimg_lat) if (x>lat_bounds[0] and x<lat_bounds[1])])
    min_lon_inds = min([i for i,x in enumerate(t_IRimg_lon) if (x>lon_bounds[0] and x<lon_bounds[1])])
    
    #take 1101 elements counting from the mins
    lat_inds = t_IRimg_lat[min_lat_inds:min_lat_inds+1700]
    lon_inds = t_IRimg_lon[min_lon_inds:min_lon_inds+2800]
    
    
    #remap the IR image
    if t_minute == 0:
        t_IRimg_remap = t_IRimg.sel(time=t_IRimg_time[0], lon = lon_inds, lat = lat_inds, method='nearest',drop=False)
        # interpolate NaN values
        t_brness = t_IRimg_remap.Tb.values
        mask = np.isnan(t_brness)
        t_brness[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), t_brness[~mask])
        t_IRimg_remap.Tb.values = t_brness
        #output
        t_IRimg_remap = t_IRimg_remap.expand_dims('time')
        t_IRimg_remap.to_netcdf(path = WORKPLACE+ r"\IRimages_remap_region" + "\\" + t_time_withmin + ".nc", mode = 'w', format = "NETCDF4" )
        print(t_time_withmin + " done")
    elif t_minute == 30: 
        t_IRimg_remap = t_IRimg.sel(time=t_IRimg_time[1], lon = lon_inds, lat = lat_inds, method='nearest')
        # interpolate NaN values
        t_brness = t_IRimg_remap.Tb.values
        mask = np.isnan(t_brness)
        t_brness[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), t_brness[~mask])
        t_IRimg_remap.Tb.values = t_brness
        #output
        t_IRimg_remap = t_IRimg_remap.expand_dims('time')
        t_IRimg_remap.to_netcdf(path = WORKPLACE+ r"\IRimages_remap_region" + "\\" + t_time_withmin + ".nc", mode = 'w', format = "NETCDF4" )
        print(t_time_withmin + " done")

#IRdataset = xr.open_dataset(r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina\IRimages_remap_interpolate_masked"+"\\" + t_time_withmin +".nc")        
#
#IRdataset_exp = IRdataset.expand_dims('time')
# =============================================================================
#%% #temporary print
 temp_brness = np.squeeze(t_IRimg_remap.Tb.values)
 temp_IRimg_lat = t_IRimg_remap.coords['lat'].values
 temp_IRimg_lon = t_IRimg_remap.coords['lon'].values
 temp_IRimg_time = t_IRimg_remap.coords['time'].values
 temp_IRimg_lat = np.round(np.squeeze(temp_IRimg_lat),2)
 temp_IRimg_lon = np.round(np.squeeze(temp_IRimg_lon),2)
 
 
 fig = plt.figure()
 im = plt.imshow(temp_brness,extent = (temp_IRimg_lon.min(),temp_IRimg_lon.max(), temp_IRimg_lat.min(),temp_IRimg_lat.max()), cmap='Greys',origin='lower',animated=True)
 cb = fig.colorbar(im, orientation='vertical')
 cb.set_label('Brightness Temperature(K)')
 
 ax = plt.gca()
 ax.set_xlabel('Longtitude')
 ax.set_ylabel('Latitude')
 plt.show()
# =============================================================================
#test_lat_inds = min([i for i,x in enumerate(t_IRimg_lat) if (x>lat_bounds[0] and x<lat_bounds[1])])
