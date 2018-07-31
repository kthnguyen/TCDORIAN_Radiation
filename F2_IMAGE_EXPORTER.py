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
IMDIR = WORKPLACE + r"\IRimages_remap_interpolate_masked_new"
#SAVDIR =  WORKPLACE + r"\Figures\180729_Labelled_data"
SAVDIR =  WORKPLACE + r"\Figures\180729_Labelled_data\cmap_fixed_270"

os.chdir(IMDIR)
files = glob.glob("2013*.nc")

#%% to save PNG
totmin = 9999
totmax = 0
for filename in files:
    #get IR image
    IRdataset = xr.open_dataset(IMDIR + "\\" + filename)
    
    t_brness = np.squeeze(IRdataset.Tb.values)
#    minval = t_brness.min()
#    maxval = t_brness.max()
#    if minval < totmin:
#        totmin = minval
#        
#    if maxval > totmax:
#        totmax = maxval
#    matplotlib.image.imsave(SAVDIR + "\\" + '2013204N11340_'+ filename.replace(".nc","")+".png", t_brness,cmap = 'Greys', origin = 'lower')
    matplotlib.image.imsave(SAVDIR + "\\" + '2013204N11340_'+ filename.replace(".nc","")+".png", t_brness,cmap = 'Greys', origin = 'lower', vmin = 170, vmax = 270)

#%%
i = 0
filename = files[i]

IRdataset = xr.open_dataset(IMDIR + "\\" + filename)

t_brness = np.squeeze(IRdataset.Tb.values)

#matplotlib.image.imsave(SAVDIR + "\\" + '2013204N11340_'+ filename.replace(".nc","")+".png", t_brness,cmap = 'Greys', origin = 'lower')

j= 84 
filename = files[j]

IRdataset = xr.open_dataset(IMDIR + "\\" + filename)

t_brness2 = np.squeeze(IRdataset.Tb.values)

j= 46 
filename = files[j]

IRdataset = xr.open_dataset(IMDIR + "\\" + filename)

t_brness3 = np.squeeze(IRdataset.Tb.values)