# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:02:00 2018

@author: z3439910
"""

import numpy as np 
import xarray as xr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from decimal import Decimal
import glob,os



WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
IMDIR = WORKPLACE + r"\IRimages_remap_interpolate_masked_new"
os.chdir(IMDIR)
files = glob.glob("2013*.nc")

# get TC estimated centers
Btracks = xr.open_dataset(WORKPLACE+"\\"+"2013204N11340.ibtracs.v03r10.nc")

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

#%%for testing
i = 0
filename = files[i]
filename = '201307310100.nc'
#get center point from best track
step = df_reindexed.iloc[i] 
t_lat = step.lat
t_lon = step.lon
i = i + 1

#get IR image
IRdataset = xr.open_dataset(IMDIR + "\\" + filename)

t_brness = np.squeeze(IRdataset.Tb.values)
t_IRimg_lat = IRdataset.coords['lat'].values
t_IRimg_lon = IRdataset.coords['lon'].values
t_IRimg_time = IRdataset.coords['time'].values
t_IRimg_lat = np.round(np.squeeze(t_IRimg_lat),2)
t_IRimg_lon = np.round(np.squeeze(t_IRimg_lon),2)

#    #mask NaN values in IR image
#    mask = np.isnan(t_brness)
#    t_brness[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), t_brness[~mask])

#plot IR image and the center point
fig = plt.figure()
im = plt.imshow(t_brness, extent = (t_IRimg_lon.min(),t_IRimg_lon.max(), t_IRimg_lat.min(),t_IRimg_lat.max()), cmap='Greys',origin='lower',animated=True)
cb = fig.colorbar(im, orientation='vertical')
cb.set_label('Brightness Temperature(K)')


ax = plt.gca()
ax.set_title('TC Dorian    '+filename.replace(".nc",""))
ax.set_xlabel('Longtitude')
ax.set_ylabel('Latitude')

#plt.plot(t_lon,t_lat,'og', markersize = 2)  

plt.show()