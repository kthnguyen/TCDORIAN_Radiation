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
IRDIR = WORKPLACE + r"\CCMP_Wind\Remap"
os.chdir(IRDIR)
files = glob.glob("TCDorian*.nc")

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

out_uwind = np.zeros(shape=(51,159,159))
out_vwind = np.zeros(shape=(51,159,159))
out_nobs = np.zeros(shape=(51,159,159))
out_wndmag = np.zeros(shape=(51,159,159))
out_wndang = np.zeros(shape=(51,159,159))
out_lat = np.zeros(shape=(51,159))
out_lon = np.zeros(shape=(51,159))
out_time = np.zeros(shape=(51))



i = 0;
#filename = files[0]
for filename in files:
    #get IR image
    IRdataset = xr.open_dataset(IRDIR + "\\" + filename)
    t_uwnd= np.squeeze(IRdataset.uwnd.values)
    t_vwnd= np.squeeze(IRdataset.vwnd.values)
    t_nobs = np.squeeze(IRdataset.nobs.values)
    t_IRimg_lat = IRdataset.coords['latitude'].values
    t_IRimg_lon = IRdataset.coords['longitude'].values
    t_IRimg_time = IRdataset.coords['time'].values
    t_IRimg_lat = np.squeeze(t_IRimg_lat)
    t_IRimg_lon = np.squeeze(t_IRimg_lon)
    
    t_wndmag = np.sqrt(np.square(t_uwnd) + np.square(t_vwnd))
    t_wndang = np.arctan(t_vwnd/t_uwnd)
    
    out_uwind[i,:,:] = t_uwnd
    out_vwind[i,:,:] = t_vwnd
    out_nobs[i,:,:] = t_nobs
    out_lat[i,:] = t_IRimg_lat
    out_lon[i,:] = t_IRimg_lon
    out_time[i] = t_IRimg_time
    out_wndmag[i,:,:] = t_wndmag
    out_wndang[i,:,:] = t_wndang

#    out_lat = np.round(out_lat,5)
#    out_lon = np.round(out_lon,5)

    i = i +1

    print (filename + " done")

outdataset = xr.Dataset({'uwnd':(['time','lon_x','lat_y'],out_uwind),\
                         'vwnd':(['time','lon_x','lat_y'],out_vwind),\
                         'nobs':(['time','lon_x','lat_y'],out_nobs),\
                         'wndmag':(['time','lon_x','lat_y'],out_wndmag),\
                         'wndang':(['time','lon_x','lat_y'],out_wndang),\
                         'lat':(['time','lat_y'],out_lat), 'lon':(['time','lon_x'], out_lon)}, coords={'time':out_time})

outdataset.to_netcdf(IRDIR + "\\" + 'Combined_TCDorian_wind.nc',mode = 'w', format = "NETCDF4")

###
#orgIRDIR = WORKPLACE + "\IR_images"
#os.chdir(orgIRDIR)
#files = glob.glob("*.nc4")
#
#filename = files[0]
#
#IRdataset = xr.open_dataset(orgIRDIR + "\\" + filename)