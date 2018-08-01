# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 09:57:39 2018

@author: z3439910
"""

#%%
t_time = t_IRimg_time.astype(np.float64)

t_time2 = pd.to_datetime(t_time)
t_time3 = np.datetime64(t_time2,'us').astype(datetime.datetime)
t_time3 = np.datetime64(t_time[0],'us')