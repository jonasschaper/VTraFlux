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
import tempfile
from scipy.sparse import diags
from scipy import interpolate
import random as rand
import warnings 
import importlib.util

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
    global script 
    global nprocs
    
    #modfold = "C:/Users/...../vtraflux_v01/" # choose directory where scripts and inpt data reside & model results are stored 
        
    par_info = {'v_ints': [24 ], 'o_int' : 24,  'v_type' : "variable", 'o_type': "constant",'offset': True, "st_t": 0, 
                'tpo': "TPO_01", "advective":"default", "reldiff":True, "T_err": 0.1, "EC_err": 0.1}  


    dat_info = {'sites' : [ "inS6_01"], 'compounds': ["EC_T"],   'chunk':(1,2),  'depths_T': ["00_20_30"],   'depthpairs_EC': ["00_19"],   'priors': 'mid'}  
    dat_info = {'sites' : [ "inS6_01"], 'compounds': ["T"],   'chunk':(1,2),  'depths_T': ["00_20_30"],   'depthpairs_EC': ["00_19"],   'priors': 'mid'}  
    dat_info = {'sites' : [ "syndata_01"], 'compounds': ["T"],   'chunk':(1,4),  'depths_T': ["00_20_30"],   'depthpairs_EC': ["00_19"],   'priors': 'mid'}  
      
    
    
    run_info = {'weights': [ 1e0,1e1], 'modfold': modfold, 'supercomputer' : False ,
            'vantage' : False,'vantage_wl': [1e-3,1e-1], 'nchains': 28 , "restart_vantage": False,
            'postrun': False, 'recover': False, 'savepars': True,  'beta0': 1.00, 'postsim': False,  'strt_opt': "rand"}


    sim_type = {'lcurve': "single", 'paramest': "lmfit", 'multicore' : True ,  'nprocs': 8,   'method': 'leastsq'}
    #sim_type = {'lcurve': "joint",  'paramest': "lmfit", 'multicore' : True,   'nprocs': 32,  'method': 'leastsq'   }
    sim_type = {'lcurve': "joint",  'paramest': "lmfit", 'multicore' : False,   'nprocs': 8,  'method': 'leastsq'   }
    
    sim_type = {'lcurve': "joint",  'paramest': "dream", 'multicore' : True,   'nprocs': 28,  'method': 'leastsq'   }
    sim_type = {'lcurve': "single",  'paramest': "dream", 'multicore' : True,   'nprocs': 28,  'method': 'leastsq'   }
    #sim_type = {'lcurve': "single",  'paramest': "dream", 'multicore' : True ,  'nprocs': 28, 'method': np.nan }



global_run_settings()

if __name__ == '__main__':

    global lcurve_est
    
    if sim_type["paramest"] == "lmfit": from lmfit import minimize, Parameters, fit_report

    nprocs = psutil.cpu_count(logical=False)
    print( 'psutil num_cpus= ',nprocs) 
    if run_info["supercomputer"]:
        nprocs = len(os.sched_getaffinity(0))  
        print( 'supercomputer num_cpus= ',nprocs)
    
        
    if sim_type["multicore"] == True:
        nprocs = sim_type["nprocs"]  
    else:
        nprocs = 1; nchains = 20


    lcurve_est = sim_type["lcurve"]
        
    if sim_type["paramest"] == "dream":
        with open(modfold +'dream_run_vtraflux.py') as file:
            script_content = file.read()
        exec(script_content) 

    if sim_type["paramest"] == "lmfit":
        with open(modfold +'lmfit_run_vtraflux.py') as file:
            script_content = file.read()
        exec(script_content) 

        
        
