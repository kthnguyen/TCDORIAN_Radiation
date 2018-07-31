# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 09:01:51 2018

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
IRDIR = WORKPLACE + r"\IRimages_remap_region"
SAVDIR = WORKPLACE + r"\Figures\080731_whole_region_labelling"
DTB = WORKPLACE + r"\Python_codes\Pickle_database"

Cdataset = xr.open_dataset(IRDIR + r"\IRDORINA_combined.nc")

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

#%%
Cflag_TC = np.zeros([Cdataset.sizes['time'],Cdataset.sizes['lon_y'],Cdataset.sizes['lat_x']])
Cflag_nonTC = np.zeros([Cdataset.sizes['time'],Cdataset.sizes['lon_y'],Cdataset.sizes['lat_x']])
Cflag_BG = np.zeros([Cdataset.sizes['time'],Cdataset.sizes['lon_y'],Cdataset.sizes['lat_x']])