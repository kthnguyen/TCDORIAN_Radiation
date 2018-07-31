# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:41:20 2018

@author: z3439910
"""

import pickle


WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\IR_Dorina"
DTB = WORKPLACE + r"\Python_codes\Pickle_database"

a = [3,4,5,6,7,8]

filename = "20180502"
outfile = open(DTB + "\\" + filename,'wb')

pickle.dump(a,outfile)
outfile.close()