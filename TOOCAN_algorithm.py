# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 00:41:49 2018

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
IRDIR = WORKPLACE + "\IRimages_remap_interpolate_masked_new"
os.chdir(IRDIR)

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

#Interpolate lat long to 0.5-hour intervals
bt = pd.DataFrame({'time':Btime,'lat':Blat,'lon':Blon})
bt = bt.set_index('time')
bt_reindexed = bt.reindex(pd.date_range(start=Btime[0],end=Btime[len(Btime)-1],freq='0.5H'))
bt_reindexed = bt_reindexed.interpolate(method='time')
bt_reindexed.index.name = 'time'
bt_reindexed.reset_index(inplace = True)

#Multilayer IR images 2013072200 till 201308040600 at 0.5h intervals
IRfile = 'output3.nc'
IRdataset = xr.open_dataset(IRDIR + "\\" + IRfile)
IR_lat = IRdataset.coords['lat'].values
IR_lon = IRdataset.coords['lon'].values
IR_time = IRdataset.coords['time'].values
IR_Tb = IRdataset.Tb.values

# plot 1 layer
laynum = 1

IRfile2 = '201307251800.nc'
IRdataset2 = xr.open_dataset(IRDIR + "\\" + IRfile2)
IR_lat2 = IRdataset2.coords['lat'].values
IR_lon2 = IRdataset2.coords['lon'].values
IR_time2 = IRdataset2.coords['time'].values
IR_Tb2 = IRdataset2.Tb.values
