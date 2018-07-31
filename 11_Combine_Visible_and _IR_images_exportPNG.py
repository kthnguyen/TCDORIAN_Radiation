# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:03:15 2018

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
from matplotlib import colors
import pickle
from matplotlib.colors import NoNorm





WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
VI_IMDIR = WORKPLACE + r"\GridSatB1_visible_IR\Remap_region"
IR_IMDIR = WORKPLACE + r"\IRimages_remap_region"
PICKLE_DTB = WORKPLACE + r"\Python_codes\Pickle_database"
os.chdir(VI_IMDIR)
vi_files = glob.glob("VIS*.nc")
os.chdir(IR_IMDIR)
ir_files = glob.glob("20130*.nc")

Cdataset = xr.open_dataset(WORKPLACE+r"\IRimages_remap_interpolate_masked_new\Combined_image_new51_theone.nc")


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

#Interpolate best track lat long to 1-hour intervals
df = pd.DataFrame({'time':Btime,'lat':Blat,'lon':Blon})
df = df.set_index('time')
df_reindexed = df.reindex(pd.date_range(start=Btime[0],end=Btime[len(Btime)-1],freq='0.5H'))
df_reindexed = df_reindexed.interpolate(method='time')
df_reindexed.index.name = 'time'
df_reindexed.reset_index(inplace = True)

#%%
Cflag = pickle.load( open( PICKLE_DTB + r"\20180503_Cflag", "rb" ))
#%% to save PNG
i = 0
j = 0
j_btrack = 0
for ir_filename in ir_files:
#for z in range(0,1):
#    ir_filename = ir_files[z]
#    
    if (i%2 == 0):
        j = int(i/2 )
    
    vi_filename = vi_files[j]
    #get center point from best track
    ir_step = df_reindexed.iloc[i] 
    ir_t_lat = ir_step.lat
    ir_t_lon = ir_step.lon
    
    vi_step = df_reindexed.iloc[2*j] 
    vi_t_lat = vi_step.lat
    vi_t_lon = vi_step.lon
    #get mask data
    c_lat = Cdataset.lat[i,:].values
    c_lon = Cdataset.lon[i,:].values
    c_Tb = Cdataset.Tb[i,:,:].values
    c_flag = Cflag[i,:,:]
    c_mask = np.where(c_flag == 0, np.NaN , c_flag)
    i = i + 1
    
    
    #get IR image
    IRdataset = xr.open_dataset(IR_IMDIR + "\\" + ir_filename)
    
    t_brness = np.squeeze(IRdataset.Tb.values)
    t_IRimg_lat = IRdataset.coords['lat'].values
    t_IRimg_lon = IRdataset.coords['lon'].values
    t_IRimg_time = IRdataset.coords['time'].values
    t_IRimg_lat = np.round(np.squeeze(t_IRimg_lat),2)
    t_IRimg_lon = np.round(np.squeeze(t_IRimg_lon),2)
    
    #get VI image
    VIdataset = xr.open_dataset(VI_IMDIR + "\\" + vi_filename)
    
    t_img = np.squeeze(VIdataset['ch1'].values)
    t_VIimg_lat = VIdataset.coords['lat'].values
    t_VIimg_lon = VIdataset.coords['lon'].values
    t_VIimg_time = VIdataset.coords['time'].values
    t_VIimg_lat = np.round(np.squeeze(t_VIimg_lat ),2)
    t_VIimg_lon = np.round(np.squeeze(t_VIimg_lat ),2)
    
    #    #mask NaN values in IR image
    #    mask = np.isnan(t_brness)
    #    t_brness[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), t_brness[~mask])
    
    #plot IR image and the center point
    fig = plt.figure()
    plt.subplot(121)
    im = plt.imshow(t_brness, extent = (t_IRimg_lon.min(),t_IRimg_lon.max(), t_IRimg_lat.min(),t_IRimg_lat.max()), cmap='gray_r',origin='lower')
    mycmap = colors.ListedColormap(['yellow'])
    im2 = plt.imshow(c_mask,extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap=mycmap,origin='lower',alpha=0.4)  
    ax = plt.gca()
    ax.set_title('TC Dorian    '+"IR_"+ir_filename.replace(".nc",""))
    ax.set_xlabel('Longtitude')
    ax.set_ylabel('Latitude')
    
    plt.plot(ir_t_lon,ir_t_lat,'or', markersize = 2)  
    
    plt.subplot(122)
    im = plt.imshow(t_img, extent = (t_IRimg_lon.min(),t_IRimg_lon.max(), t_IRimg_lat.min(),t_IRimg_lat.max()), cmap='gray',origin='lower',norm = NoNorm())
    mycmap = colors.ListedColormap(['yellow'])
    im2 = plt.imshow(c_mask,extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap=mycmap,origin='lower',alpha=0.4)  

    ax = plt.gca()
    ax.set_title('TC Dorian    '+vi_filename.replace(".nc",""))
    ax.set_xlabel('Longtitude')
    ax.set_ylabel('Latitude')
    
    plt.plot(vi_t_lon,vi_t_lat,'or', markersize = 2) 
    plt.tight_layout() 
    fig.set_size_inches(7, 3)
#    plt.show()
    fig.savefig(VI_IMDIR + r"\Combine_Figures_both_mask" + "\IR_VIS_" + ir_filename.replace(".nc","")+".png",dpi=1000)
    plt.close()
    print (ir_filename + " done")


#%%for testing
#i = 0
#j = 0

filename = files[i]
#get center point from best track
step = df_reindexed.iloc[i] 
t_lat = step.lat
t_lon = step.lon
i = i + 1

c_lat = Cdataset.lat[j,:].values
c_lon = Cdataset.lon[j,:].values
c_Tb = Cdataset.Tb[j,:,:].values
c_flag = Cflag[j,:,:]
c_mask = np.where(c_flag == 0, np.NaN , c_flag)
j = j + 2

#get IR image
IRdataset = xr.open_dataset(IMDIR + "\\" + filename)

t_brness = np.squeeze(IRdataset['ch1'].values)
t_IRimg_lat = IRdataset.coords['lat'].values
t_IRimg_lon = IRdataset.coords['lon'].values
t_IRimg_time = IRdataset.coords['time'].values
t_IRimg_lat = np.round(np.squeeze(t_IRimg_lat),2)
t_IRimg_lon = np.round(np.squeeze(t_IRimg_lon),2)

##match Cflag and regional image
#min_lat_inds = min([i for i,x in enumerate(t_IRimg_lat) if (x>(c_lat[0]-0.001) and x<(c_lat[0]+0.001))])
#min_lon_inds = min([i for i,x in enumerate(t_IRimg_lon) if (x>(c_lon[0]-0.001) and x<(c_lon[0]+0.001))])
#
#lat_inds = t_IRimg_lat[min_lat_inds:min_lat_inds+1700]
#lon_inds = t_IRimg_lon[min_lon_inds:min_lon_inds+2800]

#plot IR image and the center point
fig = plt.figure()
im = plt.imshow(t_brness, extent = (t_IRimg_lon.min(),t_IRimg_lon.max(), t_IRimg_lat.min(),t_IRimg_lat.max()), cmap='gray',origin='lower',norm = NoNorm())
mycmap = colors.ListedColormap(['yellow'])
#im2 = plt.imshow(c_mask,extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap=mycmap,origin='lower',alpha=0.2) 


ax = plt.gca()
ax.set_title('TC Dorian    '+filename.replace(".nc",""))
ax.set_xlabel('Longtitude')
ax.set_ylabel('Latitude')

plt.plot(t_lon,t_lat,'or', markersize = 2)  
fig.savefig(IMDIR + r"\Figures" + "\\" + filename.replace(".nc","")+".png",dpi=600)
plt.close()
#plt.show()
#%%fig = plt.figure()
#i = 123
#c_lat = Cdataset.lat[i,:].values
#c_lon = Cdataset.lon[i,:].values
#c_Tb = Cdataset.Tb[i,:,:].values
#c_flag = Cflag[i,:,:]
#
#fig = plt.figure()
##im = plt.imshow(c_Tb, extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap='Greys',origin='lower')
#mycmap = colors.ListedColormap(['yellow'])
#im2 = plt.imshow(c_flag,extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap=mycmap,origin='lower',alpha=0.4) 
#cb = fig.colorbar(im, orientation='vertical')
#cb.set_label('Brightness Temperature(K)')
#ax = plt.gca()
##ax.set_title('TC Dorina    '+files[C_i].replace(".nc",""))
#ax.set_xlabel('Longtitude')
#ax.set_ylabel('Latitude')
#plt.plot(t_lon,t_lat,'or', markersize = 2)  
##fig.savefig(FIGDIR + "\\" + files[C_i].replace(".nc","")+".png",dpi=600)
#plt.show()