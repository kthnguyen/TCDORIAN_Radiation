# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 23:51:07 2018

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




WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
IRDIR = WORKPLACE + r"\IRimages_remap_region"
os.chdir(IRDIR)
files = glob.glob("2013*.nc")

#outdataset = Dataset('outdataset.nc','w',format='NETCDF4_CLASSIC')
#time = outdataset.createDimension('time',3)
#lat = outdataset.createVariable('lat', np.float32, 'time')
#lon = outdataset.createVariable('lon', np.float32, 'time')
#Tb = outdataset.createVariable('Tb', np.float32, 'time')

#outdataset.description = 'The spatialtemporal image from IR images'
#outdataset.history = 'Created' + TIME.ctime(TIME.time())
#outdataset.source = 'NCEP/CPC 4km Global (60N - 60S) IR Dataset'
#
#Tb.units = 'K'

#outdataset = xr.open_dataset(IRDIR + "\\" + 'outdataset.nc')
#out_Tb = np.squeeze(outdataset.Tb.values)
#out_lat = np.squeeze(outdataset.lat.values)
#out_lon = np.squeeze(outdataset.lon.values)
file_sample = xr.open_dataset(IRDIR + "\\" + files[0])
time_dim = len(files)
lat_x_dim = np.shape(file_sample['lat'])[0]
lon_y_dim = np.shape(file_sample['lon'])[0]
out_Tb = np.zeros(shape=(time_dim,lat_x_dim,lon_y_dim))
out_lat = np.zeros(shape=(time_dim,lat_x_dim))
out_lon = np.zeros(shape=(time_dim,lon_y_dim))
out_time = np.zeros(shape=(time_dim))

#%%
i = 0;
#filename = files[0]
for filename in files:
    #get IR image
    IRdataset = xr.open_dataset(IRDIR + "\\" + filename)
    t_Tb = np.squeeze(IRdataset.Tb.values)
    t_IRimg_lat = IRdataset.coords['lat'].values
    t_IRimg_lon = IRdataset.coords['lon'].values
    t_IRimg_time = IRdataset.coords['time'].values
    t_IRimg_lat = np.squeeze(t_IRimg_lat)
    t_IRimg_lon = np.squeeze(t_IRimg_lon)
    
    out_Tb[i,:,:] = t_Tb
    out_lat[i,:] = t_IRimg_lat
    out_lon[i,:] = t_IRimg_lon
    out_time[i] = t_IRimg_time
    
#    out_lat = np.round(out_lat,5)
#    out_lon = np.round(out_lon,5)
    
    i = i +1
    
    print (filename + " done")

outdataset = xr.Dataset({'Tb':(['time','lat_x','lon_y'],out_Tb),'lat':(['time','lat_x'],out_lat),'lon':(['time','lon_y'], out_lon)}, coords={'time':out_time})

outdataset.to_netcdf(IRDIR + "\\" + 'IRDORINA_combined.nc',mode = 'w', format = "NETCDF4")

###
#orgIRDIR = WORKPLACE + "\IR_images"
#os.chdir(orgIRDIR)
#files = glob.glob("*.nc4")
#
#filename = files[0]
#
#IRdataset = xr.open_dataset(orgIRDIR + "\\" + filename)