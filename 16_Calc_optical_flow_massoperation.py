# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:08:56 2018

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
import cv2

#%% Step 1: Load directories and the combined dataset throughout the TC's life
WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
DTB = WORKPLACE + r"\Python_codes\Pickle_database"
IMDIR = WORKPLACE + r"\IRimages_remap_region"
FIGDIR = WORKPLACE + r"\Figures\Optical_flow_no_fix_point"

Cdataset = xr.open_dataset(IMDIR + r"\IRDORINA_combined.nc")

#%% Step 2: Get TC estimated centers
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

#%% Step 3: Looping through
C_i_1 = 0
C_i_2 = 1
for i in range(0,600):
    #Best track data
    step = df_reindexed.iloc[C_i_1]
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
    t_time = str_t_year + str_t_month + str_t_day + str_t_hour + str_t_minute
    # Load 2 images
    c_lat_1 = Cdataset.lat[C_i_1,:].values
    c_lon_1 = Cdataset.lon[C_i_1,:].values
    c_Tb_1 = Cdataset.Tb[C_i_1,:,:].values
    c_flag_1 = np.zeros(np.shape(c_Tb_1))
    #%
    c_lat_2 = Cdataset.lat[C_i_2,:].values
    c_lon_2 = Cdataset.lon[C_i_2,:].values
    c_Tb_2 = Cdataset.Tb[C_i_2,:,:].values
    c_flag_2 = np.zeros(np.shape(c_Tb_2))
    #%
    # Feature selection
    c_Tb_1_quant = np.uint8(c_Tb_1)
    c_Tb_1_quant[c_Tb_1_quant>290] = 0
    c_Tb_1_quant[c_Tb_1_quant>0] = 255
    
    c_Tb_2_quant = np.uint8(c_Tb_2)
    c_Tb_2_quant[c_Tb_2_quant>290] = 0
    c_Tb_2_quant[c_Tb_2_quant>0] = 255
    #plt.imshow(c_Tb_1_quant,cmap='Greys',origin='lower')
    #%
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    feature_params = dict( maxCorners = 400,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    
    corners_1 = cv2.goodFeaturesToTrack(c_Tb_1_quant,**feature_params)
    corners_1 = np.float32(corners_1)
    
#    if C_i_1 == 0:
#        corners_1 = cv2.goodFeaturesToTrack(c_Tb_1_quant,**feature_params)
#        corners_1 = np.float32(corners_1)
#    
    mask = np.zeros_like(c_Tb_1_quant)
    
    
    corners_2, st, err = cv2.calcOpticalFlowPyrLK(c_Tb_1_quant, c_Tb_2_quant, corners_1, None, **lk_params) 
    #% Plot vectors
    points = np.zeros((corners_1.shape[0],2))
    for i in range(0,corners_1.shape[0]):
        points[i,0]= corners_1[i].ravel()[0]
        points[i,1]= corners_1[i].ravel()[1]
    
    fig = plt.figure()
    im = plt.imshow(c_Tb_2,extent = [-300,3100,-300, 2000],cmap='Greys',origin='lower')
    
    #plt.imshow(img_lay, alpha = 0.2)
    plt.scatter(points[:,0],points[:,1],s = 0.1 ,c = 'g')
    
    cb = fig.colorbar(im, orientation='vertical')
    cb.set_label('Brightness Temperature(K)')
    ax = plt.gca()
    ax.set_title('TC Dorian    '+ str(t_time))
    ax.set_xlabel('Longtitude')
    ax.set_ylabel('Latitude')
#    plt.plot(t_lon*111/4,t_lat*111/4,'or', markersize = 10) 
    
    ax = plt.gca()
    ax.quiver(corners_1[:,0,0], corners_1[:,0,1], corners_2[:,0,0]-corners_1[:,0,0], corners_2[:,0,1]-corners_1[:,0,1], angles='xy', scale_units='xy', scale=0.1, width = 0.003,color = 'g')
    plt.draw()
    
    fig.set_size_inches(7, 3)
    fig.savefig(FIGDIR + "\\" + str(t_time)+".png",dpi=1000)
    plt.close()
    print (str(t_time) + "done")
#    plt.show()
    C_i_1 = C_i_1 + 1
    C_i_2 = C_i_2 + 1
#    corners_1 = corners_2[:]