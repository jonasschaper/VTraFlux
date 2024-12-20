# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:25:20 2024

@author: Jonas
"""

from regular_vtraflux import *


resultsfolder = "C:/Users/..../modres" 



directory =  os.path.join(resultsfolder, "EC_inS6_01_3_6_vv06st00_oc00_19_prib")


#find_weight(directory = directory, erfac =10,  postest = False,  mco = "MCO_02", printres = True)
find_weight(directory = directory, erfac =1,  postest = True,  mco = "MCO_02", printres = True)


#postpars, folder ,result_df= process_folders(directory = directory, filename = "current_ext.dat", postest = False) # select postest == True if dream run has finished; False if on-the-fly l curve simulation
#postpars, folder ,result_df= process_folders(directory = directory, filename = "postpars.dat", postest = True) # select postest == True if dream run has finished; False if on-the-fly l curve simulation
#smooth_l  = plt_Lcurve(subset =result_df, dirName = directory, erfac = 1, showplt = True) # estimates a smooth L-curve 
#cur_w =estimate_w(result_df, smooth_l, dirName = directory, mco = "MCO_02") # using curvature with respect to w 


