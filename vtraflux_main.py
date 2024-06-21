# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:32:30 2018

@author: schaper

"""
import os
import numpy as np
import time
import gc
import copy
import ray
import psutil
import matplotlib.pyplot as plt
import math
import copy
import pandas as pd

from scipy.sparse import diags
from scipy import interpolate
import random as rand

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from aux_vtraflux import *

def global_run_settings():

    global par_info
    global dat_info
    global run_info
    global sim_type
    global modfold
    
    modfold = "C:/Users/Jonas/Documents/tempts/transnum/vtraflux_v01a/" # choose directory where scripts and inpt data reside model results are stored 
    par_info = {'v_ints': [24 ], 'o_int' : 24,  'v_type' : "variable" , 'offset': True,    'o_type': "linear" , "st_t" : 0, 'v_int': 3}  

    dat_info = {'sites' : [ "inS6_01"],      'compounds': ["EC"], 'chunk':(1,8),  'depthpairs_T': ["NA"],   'depthpairs_EC': ["00_19"],   'priors': 'erpe'}     
    #dat_info = {'sites' : [ "erpe19_01"],    'compounds': ["EC"], 'chunk':(1,8),  'depthpairs_T': ["NA"],   'depthpairs_EC': ["00_08"],   'priors': 'erpe'}     
    #dat_info = {'sites' : [ "ammer20_01"],   'compounds': ["EC"], 'chunk':(1,8),  'depthpairs_T': ["NA"],   'depthpairs_EC': ["00_08"],   'priors': 'erpe'}     
    #dat_info = {'sites' : [ "sturt"],        'compounds': ["EC"], 'chunk':(1,8),  'depthpairs_T': ["NA"],   'depthpairs_EC': ["00_09"],   'priors': 'erpe'}     
        
    run_info = {'workstation': True, 'supercomputer' : False ,'vantage' : False,'vantage_w': 10, 'weights': [0.001,0.01,0.1,1,10,100,1000],
                'postrun': False, 'Restart': False, 'modfold': modfold, 'beta0': 1.0, 'postsim': True}

global sim_type

sim_type = {'lcurve': "single", 'estimation': "dream", 'multicore' : True  }
sim_type = {'lcurve': "joint",  'estimation': "dream", 'multicore' : True  }

global_run_settings()

if __name__ == '__main__':

    if sim_type["lcurve"] == "joint":
        cool = 'L-curve'
        # run script to estimate points of Lcurve by running multiple DREAM runs simultaneously
        with open(modfold +'vtraflux_multi.py') as file:
            script_content = file.read()
        exec(script_content) 

        
    if sim_type["lcurve"] == "single":
        cool = 'L-curve'
        # run script to initialize single DREAM run with specified weighting factor  
        with open(modfold + 'vtraflux_single.py') as file:
            script_content = file.read()
        exec(script_content) 

        
        
        
