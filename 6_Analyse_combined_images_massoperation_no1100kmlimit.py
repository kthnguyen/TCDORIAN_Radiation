# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:13:56 2018

@author: z3439910
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:03:39 2018

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
SAVDIR = WORKPLACE + r"\Figures\080730_limited_central_cold"
DTB = WORKPLACE + r"\Python_codes\Pickle_database"
os.chdir(IRDIR)
files = glob.glob("2013*.nc")

Cdataset = xr.open_dataset(IRDIR + r"\Combined_image.nc")


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

#%%VER1
@jit(['int16(float64[:,:],float64[:,:],int16,float64[:,:],int16)'])

def expand_masking(c_Tb,c_flag,Tb_thres, idx, idx_num):
    stop_flag = 1
    for i in range(0,idx_num-1):
        idx_y = idx[0][i]
        idx_x = idx[1][i]
        c_flag[idx_y,idx_x] = 2
        for jy in range (0,5):
            for jx in range (0,5):
                idx_yj = idx_y + jy
                idx_xj = idx_x + jx
                if (idx_yj>=250 and idx_yj<851 and idx_xj>=250 and idx_xj<=851 and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1
                        stop_flag = 0
                idx_yj = idx_y + jy
                idx_xj = idx_x - jx
                if (idx_yj>=250 and idx_yj<851 and idx_xj>=250 and idx_xj<=851 and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1
                        stop_flag = 0
                idx_yj = idx_y - jy
                idx_xj = idx_x + jx
                if (idx_yj>=250 and idx_yj<851 and idx_xj>=250 and idx_xj<=851 and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1
                        stop_flag = 0
                idx_yj = idx_y - jy
                idx_xj = idx_x - jx
                if (idx_yj>=250 and idx_yj<851 and idx_xj>=250 and idx_xj<=851 and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1
                        stop_flag = 0
    return stop_flag
#%%VER2
@numba.vectorize(["int16(float64,float64,int16)"], target = 'cuda')
def expand_masking(c_Tb,c_flag,Tb_thres):
    idx = np.where(c_flag==1)
#    bound = np.shape(c_Tb)
    stop_flag = 1
#    bound_y = bound[0]-1
#    bound_x = bound[1]-1

    for i in range(0,np.shape(idx)[1]-1):
        idx_y = idx[0][i]
        idx_x = idx[1][i]
        c_flag[idx_y,idx_x] = 2
        for jy in range (0,5):
            for jx in range (0,5):
                idx_yj = idx_y + jy
                idx_xj = idx_x + jx
                if (idx_yj>=250 and idx_yj<851 and idx_xj>=250 and idx_xj<=851 and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1
                        stop_flag = 0
                idx_yj = idx_y + jy
                idx_xj = idx_x - jx
                if (idx_yj>=250 and idx_yj<851 and idx_xj>=250 and idx_xj<=851 and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1
                        stop_flag = 0
                idx_yj = idx_y - jy
                idx_xj = idx_x + jx
                if (idx_yj>=250 and idx_yj<851 and idx_xj>=250 and idx_xj<=851 and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1
                        stop_flag = 0
                idx_yj = idx_y - jy
                idx_xj = idx_x - jx
                if (idx_yj>=250 and idx_yj<851 and idx_xj>=250 and idx_xj<=851 and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1
                        stop_flag = 0
    return stop_flag
#%%
@vectorize(["float32(float64,float64)"], target='cuda')
def vector_add_gpu(a, b):
    a + b
    stop_flag =1
    a.size()
    return stop_flag
#%%
#Cflag = np.zeros(np.shape(Cdataset))
Cflag = np.zeros([Cdataset.sizes['time'],Cdataset.sizes['lon_y'],Cdataset.sizes['lat_x']])
C_min_prev_mask_val = np.zeros(Cdataset.sizes['time'])
start_time_C = time.time()
for C_i in range(0,Cdataset.sizes['time']-1):
#for C_i in range(0,2):
    #Brightness temperature image
    c_lat = Cdataset.lat[C_i,:].values
    c_lon = Cdataset.lon[C_i,:].values
    c_Tb = Cdataset.Tb[C_i,:,:].values
    c_flag = np.zeros(np.shape(c_Tb))

    #Best track center
    step = df_reindexed.iloc[C_i]
    t_lat = step.lat
    t_lon = step.lon

    if C_i == 0:
        c_flag = np.where(c_Tb <= 190, 1, c_flag)
    else:
        idx_prv = np.where(Cflag[C_i-1,:,:] == 2)
        idx_prv_y = idx_prv[0]
        idx_prv_x = idx_prv[1]
        min_prv_mask_val = 9999
        min_prv_mask_idx = 0
        min_prv_mask_idy = 0
        for i in range(0,np.shape(idx_prv)[1]-1):
            if (min_prv_mask_val > c_Tb[idx_prv_y[i],idx_prv_x[i]]) & (idx_prv_x[i]>500) & (idx_prv_x[i]< 600) & (idx_prv_y[i]>500)& (idx_prv_y[i]< 600):
                min_prv_mask_val  = c_Tb[idx_prv_y[i],idx_prv_x[i]]
                min_prv_mask_idx = idx_prv_x[i]
                min_prv_mask_idy = idx_prv_y[i]

        C_min_prev_mask_val[C_i] =  min_prv_mask_val
        c_flag[min_prv_mask_idy,min_prv_mask_idx] = 1
        c_flag[min_prv_mask_idy,min_prv_mask_idx+1] = 1
        c_flag[min_prv_mask_idy,min_prv_mask_idx-1] = 1
        c_flag[min_prv_mask_idy+1,min_prv_mask_idx] = 1
        c_flag[min_prv_mask_idy-1,min_prv_mask_idx] = 1
        c_flag[min_prv_mask_idy,min_prv_mask_idx+2] = 1
        c_flag[min_prv_mask_idy,min_prv_mask_idx-2] = 1
        c_flag[min_prv_mask_idy+2,min_prv_mask_idx] = 1
        c_flag[min_prv_mask_idy-2,min_prv_mask_idx] = 1
        print("Previous mask min at value " + str(min_prv_mask_val) + " K")

    stop_flag = 0
    iteration = 1
    Tb_thres = 280
    start_time_overall = time.time()
    while stop_flag == 0:
        start_time_itr = time.time()
        stop_flag = 1
        idx = np.where(c_flag==1)
# =============================================================================
#
        bound = np.shape(c_Tb)
        bound_y = bound[0]-1
        bound_x = bound[1]-1
#
        for i in range(0,np.shape(idx)[1]-1):
                idx_y = idx[0][i]
                idx_x = idx[1][i]
                c_flag[idx_y,idx_x] = 2
                for jy in range (0,5):
                    for jx in range (0,5):
                        idx_yj = idx_y + jy
                        idx_xj = idx_x + jx
                        if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                                c_flag[idx_yj,idx_xj]=1
                                stop_flag = 0
                        idx_yj = idx_y + jy
                        idx_xj = idx_x - jx
                        if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                                c_flag[idx_yj,idx_xj]=1
                                stop_flag = 0
                        idx_yj = idx_y - jy
                        idx_xj = idx_x + jx
                        if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                                c_flag[idx_yj,idx_xj]=1
                                stop_flag = 0
                        idx_yj = idx_y - jy
                        idx_xj = idx_x - jx
                        if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                                c_flag[idx_yj,idx_xj]=1
                                stop_flag = 0
# =============================================================================
#        idx = np.where(c_flag==1)
#        idx_num = np.shape(idx)[1]-1
#        stop_flag = expand_masking(c_Tb,c_flag,Tb_thres,idx,idx_num)

        elapsed_time_itr = time.time() - start_time_itr
        print ('Layer ' + str(C_i) + ' Interation ' + str(iteration) + ' done in ' +  time.strftime("%H:%M:%S", time.gmtime(elapsed_time_itr)))
        iteration = iteration + 1

    Cflag[C_i,:,:] = c_flag

    #Plot and save figures


    c_flag = Cflag[C_i,:,:]
    c_mask = np.where(c_flag == 0, np.NaN , c_flag)
    c_Tb = Cdataset.Tb[C_i,:,:].values
    #plot IR image and the center point
    fig = plt.figure()
    im = plt.imshow(c_Tb, extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap='Greys',origin='lower')
    mycmap = colors.ListedColormap(['yellow'])
    im2 = plt.imshow(c_mask,extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap=mycmap,origin='lower',alpha=0.4)
    cb = fig.colorbar(im, orientation='vertical')
    cb.set_label('Brightness Temperature(K)')
    ax = plt.gca()
    ax.set_title('TC Dorina    '+files[C_i].replace(".nc",""))
    ax.set_xlabel('Longtitude')
    ax.set_ylabel('Latitude')
    plt.plot(t_lon,t_lat,'or', markersize = 2)
    if C_i>0:
        plt.plot(c_lon[min_prv_mask_idx],c_lat[min_prv_mask_idy],'co', markersize = 2)
    fig.savefig(SAVDIR + "\\" + files[C_i].replace(".nc","")+".png",dpi=1000)
    plt.close()
#    plt.show()
    #Print elapsed time for the current layer
    elapsed_time_overall = time.time() - start_time_overall

    print ('Cloud extraction for layer ' + str(C_i)+ ' done in ' +  time.strftime("%H:%M:%S", time.gmtime(elapsed_time_overall)))


elapsed_time_C = time.time() - start_time_C
print ('Cloud extraction for all done in ' +  time.strftime("%H:%M:%S", time.gmtime(elapsed_time_C)))
#%%
filename = "20180608_Cflag"
outfile = open(DTB + "\\" + filename,'wb')

pickle.dump(Cflag,outfile,protocol=4)
outfile.close()

#%% Plot

lay_i = 2
step = df_reindexed.iloc[lay_i]
t_lat = step.lat
t_lon = step.lon

c_flag = Cflag[lay_i,:,:]
c_mask = np.where(c_flag == 0, np.NaN , c_flag)
c_Tb = Cdataset.Tb[lay_i,:,:].values
#plot IR image and the center point
fig = plt.figure()
im = plt.imshow(c_Tb, extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap='Greys',origin='lower')
mycmap = colors.ListedColormap(['yellow'])
im2 = plt.imshow(c_mask,extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap=mycmap,origin='lower',alpha=0.4)
cb = fig.colorbar(im, orientation='vertical')
cb.set_label('Brightness Temperature(K)')
ax = plt.gca()
ax.set_title('TC Dorina    '+files[lay_i].replace(".nc",""))
ax.set_xlabel('Longtitude')
ax.set_ylabel('Latitude')

plt.plot(t_lon,t_lat,'or', markersize = 2)

#plt.show()
fig.savefig(FIGDIR + "\\" + files[lay_i].replace(".nc","")+".png",dpi=600)
plt.close()