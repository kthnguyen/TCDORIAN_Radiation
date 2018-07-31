# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 19:02:45 2018

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
import pickle




WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
IRDIR = WORKPLACE + r"\IRimages_remap_interpolate_masked_new"
DTB = WORKPLACE + r"\Pickle_files"
os.chdir(IRDIR)
files = glob.glob("*.nc")

Cdataset = xr.open_dataset(IRDIR + r"\Combined_image.nc")

#########################################################
#%% Plot layer
i = 0

c_lat = Cdataset.lat[i,:].values
c_lon = Cdataset.lon[i,:].values
c_Tb = Cdataset.Tb[i,:,:].values

c_mask = np.where(c_Tb > 190, np.NaN , c_Tb)

#plot IR image and the center point
fig = plt.figure()
im = plt.imshow(c_Tb, extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap='Greys',origin='lower')
im2 = plt.imshow(c_mask,extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap='Oranges',origin='lower') 
cb = fig.colorbar(im, orientation='vertical')
cb.set_label('Brightness Temperature(K)')
plt.show()
#########################################################
#%% Process the 1st layer
c_flag = np.zeros(np.shape(c_Tb))
c_flag = np.where(c_Tb <= 190, 1, c_flag)
#fig = plt.figure()
#im = plt.imshow(c_flag, origin='lower')
#plt.show()

stop_flag = False
iteration = 1
Tb_thres = 270
start_time_overall = time.time()
while stop_flag == False:
    start_time_itr = time.time()
    stop_flag = True
    idx = np.where(c_flag==1)
    bound = np.shape(c_Tb)
    bound_y = bound[0]-1
    bound_x = bound[1]-1
    
    for i in range(0,np.shape(idx)[1]-1):
        idx_y = idx[0][i]
        idx_x = idx[1][i]
        c_flag[idx_y,idx_x] = 2
        for jy in range (0,10):
            for jx in range (0,10): 
                idx_yj = idx_y + jy
                idx_xj = idx_x + jx
                if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1 
                        stop_flag = False    
                idx_yj = idx_y + jy
                idx_xj = idx_x - jx
                if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1 
                        stop_flag = False  
                idx_yj = idx_y - jy
                idx_xj = idx_x + jx
                if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1 
                        stop_flag = False  
                idx_yj = idx_y - jy
                idx_xj = idx_x - jx
                if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb[idx_yj,idx_xj]<=Tb_thres) and c_flag[idx_yj,idx_xj]==0):
                        c_flag[idx_yj,idx_xj]=1 
                        stop_flag = False  
    
    elapsed_time_itr = time.time() - start_time_itr
    print ('Interation ' + str(iteration) + ' done in ' +  time.strftime("%H:%M:%S", time.gmtime(elapsed_time_itr)))
    iteration = iteration + 1

elapsed_time_overall = time.time() - start_time_overall
print ('Cloud extraction done in ' +  time.strftime("%H:%M:%S", time.gmtime(elapsed_time_overall)))
#%%
#plot IR image and the center point
c_mask = np.where(c_flag == 0, np.NaN , c_flag)
fig = plt.figure()
im = plt.imshow(c_Tb, extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap='Greys',origin='lower')
im2 = plt.imshow(c_mask,extent = (c_lon.min(),c_lon.max(), c_lat.min(),c_lat.max()), cmap='plasma',origin='lower', alpha = 0.4) 
cb = fig.colorbar(im, orientation='vertical')
cb.set_label('Brightness Temperature(K)')
plt.show()

#%% Layer 2
i = 1

c_lat2 = Cdataset.lat[i,:].values
c_lon2 = Cdataset.lon[i,:].values
c_Tb2 = Cdataset.Tb[i,:,:].values


#plot IR image and the center point
c_mask = np.where(c_flag == 0, np.NaN , c_flag)
fig = plt.figure()
im = plt.imshow(c_Tb2, extent = (c_lon2.min(),c_lon2.max(), c_lat2.min(),c_lat2.max()), cmap='Greys',origin='lower')
mycmap = colors.ListedColormap(['yellow'])
im2 = plt.imshow(c_mask,extent = (c_lon2.min(),c_lon2.max(), c_lat2.min(),c_lat2.max()), cmap=mycmap,origin='lower', alpha = 0.4) 
cb = fig.colorbar(im, orientation='vertical')
cb.set_label('Brightness Temperature(K)')
plt.show()

#%% Extract from Layer 1 mask to Layer 2
c_flag2 = np.zeros(np.shape(c_Tb2))

idx_prv = np.where(c_flag == 2)
idx_prv_y = idx_prv[0]
idx_prv_x = idx_prv[1]
for i in range(0,np.shape(idx_prv)[1]-1):
    if (c_Tb2[idx_prv_y[i],idx_prv_x[i]] <=270):
        c_flag2[idx_prv_y[i],idx_prv_x[i]] = 1

stop_flag = False
iteration = 1
Tb_thres = 270
start_time_overall = time.time()
while stop_flag == False:
    start_time_itr = time.time()
    stop_flag = True
    idx = np.where(c_flag2==1)
    bound = np.shape(c_Tb2)
    bound_y = bound[0]-1
    bound_x = bound[1]-1
    
    for i in range(0,np.shape(idx)[1]-1):
        idx_y = idx[0][i]
        idx_x = idx[1][i]
        c_flag2[idx_y,idx_x] = 2
        for jy in range (0,5):
            for jx in range (0,5): 
                idx_yj = idx_y + jy
                idx_xj = idx_x + jx
                if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb2[idx_yj,idx_xj]<=Tb_thres) and c_flag2[idx_yj,idx_xj]==0):
                        c_flag2[idx_yj,idx_xj]=1 
                        stop_flag = False    
                idx_yj = idx_y + jy
                idx_xj = idx_x - jx
                if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb2[idx_yj,idx_xj]<=Tb_thres) and c_flag2[idx_yj,idx_xj]==0):
                        c_flag2[idx_yj,idx_xj]=1 
                        stop_flag = False  
                idx_yj = idx_y - jy
                idx_xj = idx_x + jx
                if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb2[idx_yj,idx_xj]<=Tb_thres) and c_flag2[idx_yj,idx_xj]==0):
                        c_flag2[idx_yj,idx_xj]=1 
                        stop_flag = False  
                idx_yj = idx_y - jy
                idx_xj = idx_x - jx
                if (idx_yj>=0 and idx_yj<bound_y and idx_xj>=0 and idx_xj<=bound_x and (c_Tb2[idx_yj,idx_xj]<=Tb_thres) and c_flag2[idx_yj,idx_xj]==0):
                        c_flag2[idx_yj,idx_xj]=1 
                        stop_flag = False  
    
    elapsed_time_itr = time.time() - start_time_itr
    print ('Interation ' + str(iteration) + ' done in ' +  time.strftime("%H:%M:%S", time.gmtime(elapsed_time_itr)))
    iteration = iteration + 1

elapsed_time_overall = time.time() - start_time_overall
print ('Cloud extraction done in ' +  time.strftime("%H:%M:%S", time.gmtime(elapsed_time_overall)))

#%%
#plot IR image and the center point
c_mask2 = np.where(c_flag2 == 0, np.NaN , c_flag2)
fig = plt.figure()
im = plt.imshow(c_Tb2, extent = (c_lon2.min(),c_lon2.max(), c_lat2.min(),c_lat2.max()), cmap='Greys',origin='lower')
mycmap = colors.ListedColormap(['yellow'])
im2 = plt.imshow(c_mask2,extent = (c_lon2.min(),c_lon2.max(), c_lat2.min(),c_lat2.max()), cmap=mycmap,origin='lower', alpha = 0.4) 
cb = fig.colorbar(im, orientation='vertical')
cb.set_label('Brightness Temperature(K)')
plt.show()

#%%
filename = "20180503_Cflag"
outfile = open(DTB + "\\" + filename,'wb')

pickle.dump(Cflag,outfile,protocol=4)
outfile.close()