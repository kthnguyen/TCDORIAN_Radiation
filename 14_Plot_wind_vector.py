# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:31:12 2018

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
import time
from matplotlib import colors
import numba
from numba import vectorize, float64, int16, guvectorize, jit
import pickle
import iris
import iris.coord_categorisation
import iris.quickplot as qplt

WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
IMDIR = WORKPLACE + r"\CCMP_Wind\Remap"
FIGDIR = IMDIR + r"\Figures"
DTB = WORKPLACE + r"\Python_codes\Pickle_database"
os.chdir(IMDIR)
files = glob.glob("TCDorian*.nc")

Wdataset = xr.open_dataset(IMDIR + r"\Combined_TCDorian_wind.nc")

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
#%% Plot just vector
for W_i in range(0,51):
    #Best track data
    step = df_reindexed.iloc[W_i]
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
    
    ## Wind data
    w_lat = Wdataset.lat[W_i,:].values
    w_lon = Wdataset.lon[W_i,:].values
    w_uwnd = Wdataset.uwnd[W_i,:].values
    w_vwnd = Wdataset.vwnd[W_i,:].values
    w_wndmag = Wdataset.wndmag[W_i,:].values
    
    convert_coords(w_lon,"to180")
    

    
    w_uwnd_norm = w_uwnd / np.sqrt(w_uwnd ** 2.0 + w_vwnd ** 2.0)
    w_vwnd_norm = w_vwnd / np.sqrt(w_uwnd ** 2.0 + w_vwnd ** 2.0)
   
    ##PLOT
    fig = plt.figure()
    SMALL_SIZE = 6
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
#    qplt.contourf(w_wndmag,20)
    X,Y = np.meshgrid(w_lon,w_lat)
    skip = (slice(None, None, 3), slice(None, None, 3))
    plt.quiver(X[skip], Y[skip], w_uwnd[skip],w_vwnd[skip],scale_units='x',scale = 10)
    im = plt.imshow(w_wndmag*1.94, extent = (w_lon.min(),w_lon.max(), w_lat.min(),w_lat.max()), cmap='coolwarm',origin='lower',animated=True)
    plt.plot(t_lon,t_lat,'go', markersize = 2) 
    
    cb = fig.colorbar(im, orientation='horizontal',fraction=0.04, pad=0.1)
    cb.set_label(r'SS+10m Wind Speed (knot)')
    ax = plt.gca()
    ax.set_xlabel('Longtitude')
    ax.set_ylabel('Latitude')
    ax.set_title('TC Dorian ' + str(t_time) + r" Wind Vector")
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    
    fig.set_size_inches(7, 3)
    fig.savefig(IMDIR + r"\Figures\TCDorian_windvector_" + str(t_time)+".png",dpi=1000)
    plt.close()
#    plt.show()
    
#%% Plot u,v and vector
for W_i in range(0,51):
    #Best track data
    step = df_reindexed.iloc[W_i]
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
    
    ## Wind data
    w_lat = Wdataset.lat[W_i,:].values
    w_lon = Wdataset.lon[W_i,:].values
    w_uwnd = Wdataset.uwnd[W_i,:].values
    w_vwnd = Wdataset.vwnd[W_i,:].values
    w_wndmag = Wdataset.wndmag[W_i,:].values
    
    convert_coords(w_lon,"to180")
    
    
    
    w_uwnd_norm = w_uwnd / np.sqrt(w_uwnd ** 2.0 + w_vwnd ** 2.0)
    w_vwnd_norm = w_vwnd / np.sqrt(w_uwnd ** 2.0 + w_vwnd ** 2.0)
       
    ##PLOT
    fig = plt.figure()
    plt.subplot(131)
    im = plt.imshow(w_uwnd*1.94, extent = (w_lon.min(),w_lon.max(), w_lat.min(),w_lat.max()), cmap='coolwarm',origin='lower',animated=True)
    cb = fig.colorbar(im, orientation='horizontal',)
    cb.set_label(r'SS+10m Wind speed (knot)')
    
    #plt.plot(t_lon+360,t_lat,'go', markersize = 2)  
    ax = plt.gca()
    ax.set_xlabel('Longtitude')
    ax.set_ylabel('Latitude')
    ax.set_title('TC Dorian ' + t_time + "00" + r" Uwind (eastward)")
    
    #######################
    plt.subplot(132)
    im = plt.imshow(w_vwnd*1.94, extent = (w_lon.min(),w_lon.max(), w_lat.min(),w_lat.max()), cmap='coolwarm',origin='lower',animated=True)
    cb = fig.colorbar(im, orientation='horizontal')
    cb.set_label(r'SS+10m Wind speed (knot)')
    
    #plt.plot(t_lon+360,t_lat,'go', markersize = 2)  
    ax = plt.gca()
    ax.set_xlabel('Longtitude')
    ax.set_ylabel('Latitude')
    ax.set_title('TC Dorian ' + t_time + "00" + r" Vwind (northward)")
    
    #######################
    plt.subplot(133)
    SMALL_SIZE = 6
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    #    qplt.contourf(w_wndmag,20)
    X,Y = np.meshgrid(w_lon,w_lat)
    skip = (slice(None, None, 3), slice(None, None, 3))
    plt.quiver(X[skip], Y[skip], w_uwnd[skip],w_vwnd[skip],scale_units='x',scale = 8)
    im = plt.imshow(w_wndmag*1.94, extent = (w_lon.min(),w_lon.max(), w_lat.min(),w_lat.max()), cmap='coolwarm',origin='lower',animated=True)
    plt.plot(t_lon,t_lat,'go', markersize = 2) 
    
    cb = fig.colorbar(im, orientation='horizontal',fraction=0.04, pad=0.1)
    cb.set_label(r'SS+10m Wind Speed (knot)')
    ax = plt.gca()
    ax.set_xlabel('Longtitude')
    ax.set_ylabel('Latitude')
    ax.set_title('TC Dorian ' + str(t_time) + r" Wind Vector")
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    
    fig.set_size_inches(9, 3)
    fig.savefig(IMDIR + r"\Figures\Wind_vector\TCDorian_windvector_" + str(t_time)+".png",dpi=1000)
    plt.close()
    #plt.show()    
    
#%% Test single image
W_i = 0
#Best track data
step = df_reindexed.iloc[W_i]
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

## Wind data
w_lat = Wdataset.lat[W_i,:].values
w_lon = Wdataset.lon[W_i,:].values
w_uwnd = Wdataset.uwnd[W_i,:].values
w_vwnd = Wdataset.vwnd[W_i,:].values
w_wndmag = Wdataset.wndmag[W_i,:].values

convert_coords(w_lon,"to180")



w_uwnd_norm = w_uwnd / np.sqrt(w_uwnd ** 2.0 + w_vwnd ** 2.0)
w_vwnd_norm = w_vwnd / np.sqrt(w_uwnd ** 2.0 + w_vwnd ** 2.0)
   
##PLOT
fig = plt.figure()
plt.subplot(131)
im = plt.imshow(w_uwnd, extent = (w_lon.min(),w_lon.max(), w_lat.min(),w_lat.max()), cmap='coolwarm',origin='lower',animated=True)
cb = fig.colorbar(im, orientation='horizontal',)
cb.set_label(r'SS+10m Wind speed (m/s)')

#plt.plot(t_lon+360,t_lat,'go', markersize = 2)  
ax = plt.gca()
ax.set_xlabel('Longtitude')
ax.set_ylabel('Latitude')
ax.set_title('TC Dorian ' + t_time + "00" + r" Uwind (eastward)")

#######################
plt.subplot(132)
im = plt.imshow(w_vwnd, extent = (w_lon.min(),w_lon.max(), w_lat.min(),w_lat.max()), cmap='coolwarm',origin='lower',animated=True)
cb = fig.colorbar(im, orientation='horizontal')
cb.set_label(r'SS+10m Wind speed (m/s)')

#plt.plot(t_lon+360,t_lat,'go', markersize = 2)  
ax = plt.gca()
ax.set_xlabel('Longtitude')
ax.set_ylabel('Latitude')
ax.set_title('TC Dorian ' + t_time + "00" + r" Vwind (northward)")

#######################
plt.subplot(133)
SMALL_SIZE = 6
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
#    qplt.contourf(w_wndmag,20)
X,Y = np.meshgrid(w_lon,w_lat)
skip = (slice(None, None, 3), slice(None, None, 3))
plt.quiver(X[skip], Y[skip], w_uwnd[skip],w_vwnd[skip],scale_units='x',scale = 8)
im = plt.imshow(w_wndmag*1.94, extent = (w_lon.min(),w_lon.max(), w_lat.min(),w_lat.max()), cmap='coolwarm',origin='lower',animated=True)
plt.plot(t_lon,t_lat,'go', markersize = 2) 

cb = fig.colorbar(im, orientation='horizontal',fraction=0.04, pad=0.1)
cb.set_label(r'SS+10m Wind Speed (knot)')
ax = plt.gca()
ax.set_xlabel('Longtitude')
ax.set_ylabel('Latitude')
ax.set_title('TC Dorian ' + str(t_time) + r" Wind Vector")

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

fig.set_size_inches(9, 3)
fig.savefig(IMDIR + r"\Figures\Wind_vector\TCDorian_windvector_" + str(t_time)+".png",dpi=1000)
plt.close()
#plt.show()