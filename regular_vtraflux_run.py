# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:25:20 2024

@author: Jonas
"""

from regular_vtraflux import *
secondraft = "path where DREAM result - folders are located " 

directory =  os.path.join(secondraft,"sturt", "EC_sturt_1_8_vv03st00_oc00_9_prie")
postpars, folder ,result_df= process_folders(directory = directory, filename = "postpars.dat", postest = True) # select postest == True if dream run has finished; False if on-the-fly l curve simulation
smooth_l  = plt_Lcurve(subset =result_df,dirName = directory, erfac = 10, showplt = True) # estimates a smooth L-curve 
cur_w =estimate_w(result_df, smooth_l, dirName = directory) # using curvature with respect to w 
