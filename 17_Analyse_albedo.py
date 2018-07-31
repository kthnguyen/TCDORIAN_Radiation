# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:08:21 2018

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



WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
IRDIR = WORKPLACE + r"\IRimages_remap_interpolate_masked_new"
FIGDIR = WORKPLACE + r"\Figures"
DTB = WORKPLACE + r"\Python_codes\Pickle_database"
PICKLE_DTB = WORKPLACE + r"\Python_codes\Pickle_database"
ALBEDO_DTB = WORKPLACE + r"\CERESdata\CERES_201307_08"
os.chdir(IRDIR)
files = glob.glob("2013*.nc")

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
def sum1(input):
    return sum(map(sum, input))
#%%
Cdataset = xr.open_dataset(IRDIR + r"\Combined_image.nc")
Cflag = pickle.load( open(PICKLE_DTB + r"\20180608_Cflag", "rb" ))
#%%
Adataset = xr.open_dataset(ALBEDO_DTB + r"\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4A_Subset_20130701-20130831.nc")
#%%
Atime = Adataset['time'].values
#Adataset[(Adataset['time'] > '2013-07-22') & (Adataset['time'] < '2013-08-04')]
#Adataset[Adataset['time'].isin(pd.date_range('2013-07-22','2013-08-04'))]
#%% Best track

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

#%% FIRST ANALYSIS
C_i = 1 + 48*4
c_lat = Cdataset.lat[C_i,:].values
c_lon = Cdataset.lon[C_i,:].values
c_Tb = Cdataset.Tb[C_i,:,:].values
c_time = pd.to_datetime(Cdataset['time'].values[C_i])
c_flag = Cflag[C_i,:,:]
convert_coords(c_lon, "to360")
#%
A_i = 522 + 24*4
a_sw = Adataset.toa_sw_all_1h[A_i,:,:].values
a_lw = Adataset.toa_lw_all_1h[A_i,:,:].values
a_lat = Adataset['lat'].values
a_lon = Adataset['lon'].values

#%
c_flag_pos = np.where(c_flag>0)
c_flag_pos_lat = c_lat[c_flag_pos[0]]
c_flag_pos_lon = c_lon[c_flag_pos[1]]


#%
mask_sw = 0
mask_lw = 0
for CA_i in range(0, len(c_flag_pos_lat)-1):
    sel_lat = c_flag_pos_lat[CA_i]
    sel_lon = c_flag_pos_lon[CA_i]
    sel_lat_idx = max(np.where((a_lat<sel_lat))[0])
    sel_lon_idx = max(np.where((a_lon<sel_lon))[0])
    sel_sw = a_sw[sel_lat_idx,sel_lon_idx]
    sel_lw = a_lw[sel_lat_idx,sel_lon_idx]
    mask_sw += sel_sw 
    mask_lw += sel_lw

sum_sw = sum1(a_sw)
sum_lw = sum1(a_lw)

mask_pc_sw = (mask_sw*16)/(sum_sw*12321)
mask_pc_lw = (mask_lw*16)/(sum_lw*12321)
#%
fig = plt.figure()
im = plt.imshow(a_sw)
plt.show()
#%% FULL ANALYSIS
i = 0
C_i = 1
mask_pc_sw = np.zeros(301)
mask_pc_lw = np.zeros(301)
mask_sw_dur = np.zeros(301)
mask_lw_dur = np.zeros(301) 
sum_sw_dur = np.zeros(301) 
sum_lw_dur = np.zeros(301) 
while C_i < 600:
    c_lat = Cdataset.lat[C_i,:].values
    c_lon = Cdataset.lon[C_i,:].values
    c_Tb = Cdataset.Tb[C_i,:,:].values
    c_time = pd.to_datetime(Cdataset['time'].values[C_i])
    c_flag = Cflag[C_i,:,:]
    convert_coords(c_lon, "to360")
    #%
    A_i = 522 + i
    a_sw = Adataset.toa_sw_all_1h[A_i,:,:].values
    a_lw = Adataset.toa_lw_all_1h[A_i,:,:].values
    a_lat = Adataset['lat'].values
    a_lon = Adataset['lon'].values
    
    #%
    c_flag_pos = np.where(c_flag>0)
    c_flag_pos_lat = c_lat[c_flag_pos[0]]
    c_flag_pos_lon = c_lon[c_flag_pos[1]]
    
    
    #%
    mask_sw = 0
    mask_lw = 0
    for CA_i in range(0, len(c_flag_pos_lat)-1):
        sel_lat = c_flag_pos_lat[CA_i]
        sel_lon = c_flag_pos_lon[CA_i]
        sel_lat_idx = max(np.where((a_lat<sel_lat))[0])
        sel_lon_idx = max(np.where((a_lon<sel_lon))[0])
        sel_sw = a_sw[sel_lat_idx,sel_lon_idx]
        sel_lw = a_lw[sel_lat_idx,sel_lon_idx]
        mask_sw += sel_sw 
        mask_lw += sel_lw
    
    sum_sw = sum1(a_sw)
    sum_lw = sum1(a_lw)
    
    sum_sw_dur[i] = sum_sw
    sum_lw_dur[i] = sum_lw
    mask_sw_dur[i] = mask_sw
    mask_lw_dur[i] = mask_lw
    mask_pc_sw[i] = (mask_sw*16)/(sum_sw*12321)
    mask_pc_lw[i] = (mask_lw*16)/(sum_lw*12321)
    i = i + 1
    C_i = C_i + 2
    print(str(c_time) + " done")
#%% Overall
mask_pc_sw_dur = (sum(mask_sw_dur)*16)/(sum(sum_sw_dur)*12321)
print ("Percentage of shortwave contribution in the NA Ocean during TC life: " + str(mask_pc_sw_dur*100)+ " percent")
mask_pc_lw_dur = (sum(mask_lw_dur)*16)/(sum(sum_lw_dur)*12321)
print ("Percentage of longwave contribution in the NA Ocean during TC life: " + str(mask_pc_lw_dur*100)+ " percent")
#%%
fig = plt.figure()
plt.plot(mask_pc_sw*100)
ax = plt.gca()
ax.set_title('TC Dorian - Shortwave Contribution in the NA Ocean')
ax.set_xlabel('2018/07/22 183000h - 2018/08/04 063000h')
ax.set_ylabel('Percent')             
fig.savefig(FIGDIR + "\DORIAN_SW.png",dpi=1000)

fig = plt.figure()
plt.plot(mask_pc_lw*100)
ax = plt.gca()
ax.set_title('TC Dorian - Longwave Contribution in the NA Ocean')
ax.set_xlabel('2018/07/22 183000h - 2018/08/04 063000h')
ax.set_ylabel('Percent')             
fig.savefig(FIGDIR  + "\DORIAN_LW.png",dpi=1000)
