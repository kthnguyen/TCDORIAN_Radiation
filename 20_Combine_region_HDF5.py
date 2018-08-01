# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:22:56 2018

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
from netCDF4 import Dataset
import time as TIME
import h5py
from datetime import datetime




WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
IRDIR = WORKPLACE + r"\IRimages_remap_region"
os.chdir(IRDIR)
files = glob.glob("2013*.nc")

#%%
Hfile = h5py.File('TCDORIAN.h5','w')
Hfile.create_dataset('time',shape = (601,1),dtype = np.float64)
i = 0;
#filename = files[0]
for filename in files:
    #get IR image
    IRdataset = xr.open_dataset(IRDIR + "\\" + filename)
    t_Tb = np.squeeze(IRdataset.Tb.values)
    t_IRimg_lat = IRdataset.coords['lat'].values
    t_IRimg_lon = IRdataset.coords['lon'].values
    t_IRimg_time = IRdataset.coords['time'].values
    #t_IRimg_lat = np.squeeze(t_IRimg_lat)
    #t_IRimg_lon = np.squeeze(t_IRimg_lon)
    Hfile.create_dataset(filename.replace(".nc",""),data = t_Tb)
    if i == 0:
        Hfile.create_dataset('longitude',data = t_IRimg_lon)
        Hfile.create_dataset('latitude',data = t_IRimg_lat)
    t_time = t_IRimg_time.astype(np.float64)
    Hfile['time'][i] = t_time
    i+=1
    
    print (filename + " done")
Hfile.close()
