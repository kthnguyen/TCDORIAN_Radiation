# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:45:15 2018

@author: z3439910
"""

import pickle


WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
DTB = WORKPLACE + r"\Python_codes\Pickle_database"

filename = "20180502"
infile = open(DTB + "\\" + filename,'rb')
b = pickle.load(infile)
infile.close()