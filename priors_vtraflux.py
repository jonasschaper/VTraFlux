# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:13:33 2024

@author: Jonas Schaper
"""

import numpy as np

####################################################################################################################################################################################### 
#######################################################################################################################################################################################

def prior_range_get(prange):
    

    if prange == "erpe":
    
        priors = { 'vv':    {'min': 0.0025, 'max': 0.15, 'unif': True,  'log': False , 'mea': np.nan,  'ste': np.nan, 'strt': 0 },
                   'al':    {'min': 0.005,  'max': 0.1, 'unif': True,  'log': False , 'mea': np.nan,  'ste': np.nan , 'strt': 0},
                   'offset': {'min': -0.1,   'max': 0.1,  'unif': True,  'log': False  , 'mea': np.nan,  'ste': np.nan , 'strt': 0 }
                   }   
  
    if prange == "mid":
        priors = { 'vv':    {'min': 0.0001, 'max': 0.40, 'unif': True,  'log': False , 'strt': 0},
                   'al':    {'min': 0.005,  'max': 0.15, 'unif': True,  'log': False , 'strt': 0},
                   'offset': {'min': -0.1,   'max': 0.1,  'unif': True,  'log': False , 'strt': 0  },
                   'beta':  {'min': 0.0001, 'max': 0.1, 'unif': True , 'log': False , 'strt': 0 },
                   'pcs':  {'min': 1.5*1e6,      'max': 3.4*1e6 ,  'unif': True  ,'log': False , 'strt': 0 },
                   'ks':   {'min': 2.4*3600,      'max': 8.4*3600 ,  'unif': True , 'log': False , 'strt': 0 }, # is ksppa_s thermal conductivty of solids in W m-1 C-1 
                   'n':     {'min': 0.25,      'max': 0.55,   'unif': True,  'log': False, 'strt': 0   },
                   'offset':   {'min': -0.1, 'max': 0.1,   'unif': True,  'log': False, 'strt': 0   }}
        
                 
    if prange == "wide":
        # Stonestrom, D. A., and K. W. Blasch (2003), Determining temperature and thermal properties for heat-based studies of surface-water ground-water interactions, in Heat as a Tool for Studying the Movement of Ground
        priors = { 'vv':    {'min': -0.40, 'max': 0.40, 'unif': True,  'log': False, 'strt': 0 },
                   'al':    {'min': 0.005,  'max': 0.15, 'unif': True,  'log': False, 'strt': 0 },
                   'offset': {'min': -0.1,   'max': 0.1,  'unif': True,  'log': False , 'strt': 0  },
                   #'beta':  {'min': 0.0005, 'max': 0.1, 'unif': True , 'log': False  },
                   'beta':  {'min': 0.0001, 'max': 0.1, 'unif': True , 'log': False , 'strt': 0 },
                   'pcs':  {'min': 1.5*1e6,      'max': 3.4*1e6 ,  'unif': True  ,'log': False , 'strt': 0 },
                   'ks':   {'min': 2.4*3600,      'max': 8.4*3600 ,  'unif': True , 'log': False , 'strt': 0 }, # is ksppa_s thermal conductivty of solids in W m-1 C-1 
                   'n':     {'min': 0.25,      'max': 0.55,   'unif': True,  'log': False  , 'strt': 0 },
                   'offset':   {'min': -0.1, 'max': 0.1,   'unif': True,  'log': False , 'strt': 0  }}
                               
    
    if prange == "dupdow_lbhk":
        priors = { 'vv':    {'min': -0.40, 'max': 0.40, 'unif': True,  'log': False, 'strt': 0 },
                   'al':    {'min': 0.005,  'max': 0.15, 'unif': True,  'log': False , 'strt': 0},
                   'offset': {'min': -0.1,   'max': 0.1,  'unif': True,  'log': False , 'strt': 0  },
                   'beta':  {'min': 0.0001, 'max': 0.005, 'unif': True , 'log': False , 'strt': 0 },
                   'pcs':  {'min': 1.5*1e6,      'max': 3.4*1e6 ,  'unif': True  ,'log': False , 'strt': 0 },
                   'ks':   {'min': 2.4*3600,      'max': 8.4*3600 ,  'unif': True , 'log': False , 'strt': 0 }, # is ksppa_s thermal conductivty of solids in W m-1 C-1 
                   'n':     {'min': 0.25,      'max': 0.55,   'unif': True,  'log': False  , 'strt': 0 },
                   'offset':   {'min': -0.1, 'max': 0.1,   'unif': True,  'log': False  , 'strt': 0 }}
                           
    
    if prange == "bdown_mbhk":
        priors = { 'vv':    {'min': 0.0001, 'max': 0.40, 'unif': True,  'log': False , 'strt': 0},
               'al':    {'min': 0.005,  'max': 0.15, 'unif': True,  'log': False, 'strt': 0 },
               'offset': {'min': -0.1,   'max': 0.1,  'unif': True,  'log': False , 'strt': 0  },
               'beta':  {'min': 0.0001, 'max': 0.025, 'unif': True , 'log': False , 'strt': 0 },
               'pcs':  {'min': 1.5*1e6,      'max': 3.4*1e6 ,  'unif': True  ,'log': False, 'strt': 0  },
               'pcb':  {'min': 1.5*1e6,      'max': 3.4*1e6 ,  'unif': True  ,'log': False  , 'strt': 0},
               'ks':   {'min': 2.4*3600,      'max':  8.4*3600 ,  'unif': True , 'log': False  , 'strt': 0}, # kappa_s thermal conductivty of solids in W m-1 C-1 
               'ke':   {'min': 0.05/24,      'max':  0.1/24 ,  'unif': True , 'log': False  , 'strt': 0}, #  thermal diffusivity of solids in W m-1 C-1 
               'yt':   {'min': 0.3,      'max':  0.75 ,  'unif': True , 'log': False  , 'strt': 0}, # is kappa_s thermal conductivty of solids in W m-1 C-1 
               'k0':   {'min': 2.4*3600,      'max':  3.4*3600 ,  'unif': True , 'log': False , 'strt': 0 }, # is kappa_s thermal conductivty of solids in W m-1 C-1 
               'n':     {'min': 0.25,      'max': 0.55,   'unif': True,  'log': False  , 'strt': 0 },
               'offset':   {'min': -0.1, 'max': 0.1,   'unif': True,  'log': False  , 'strt': 0 }}

    
    print('prior range: ', prange)
    return priors


#######################################################################################################################################################################################
#######################################################################################################################################################################################


def load_priors(ndays, comps,param_info, prange):
    # values (TPO 1, compare section 2.1.4).  γ_T and κ_e by specifying prior ranges of sediment porosity, bulk volumetric heat capacity ρ_b c_b and the bulk thermal conductivity κ_0 (TPO 2)
    # and allows users to directly specify prior ranges of γ_T and κ_e (TPO 3). TPO 2 is intended to be used in situations, in which ρ_b c_b and κ_0 have been measured from sediment samples by, for instance, KD2Pro device (Decagon Devices, Inc.). 
    
    priors = prior_range_get(prange)
    
    unif  = [];     Mean  = [];     strt  = []
    width = [];     rmin  = [];     rmax  = []
    # unifrom conservative parameter priors 
    
    if "EC" in comps: 
        unif,Mean,width,rmin,rmax, strt = prior_make(priors,'al',unif,Mean,width,rmin,rmax, strt)    
 
    if param_info["v_type"] == "variable":
        ittt = 1
        if param_info["st_t"] == 0: ittt = 0
        for i in range(int( (ndays*24/param_info["v_int"])+1 - ((param_info["st_t"]/param_info["v_int"])-ittt) )): 
                    unif,Mean,width,rmin,rmax, strt = prior_make(priors,'vv',unif,Mean,width,rmin,rmax, strt)
    else:
        unif,Mean,width,rmin,rmax, strt = prior_make(priors,'vv',unif,Mean,width,rmin,rmax, strt)

    if "EC" in comps and param_info["offset"] == True:     
        if param_info["o_type"] == "constant":
            unif,Mean,width,rmin,rmax, strt = prior_make(priors,'offset',unif,Mean,width,rmin,rmax, strt)
        if param_info["o_type"] == "variable":
            for i in range(int(ndays*24/param_info["o_int"]+1)): 
                    unif,Mean,width,rmin,rmax, strt = prior_make(priors,'offset',unif,Mean,width,rmin,rmax, strt)
        if param_info["o_type"] == "linear":
                    unif,Mean,width,rmin,rmax, strt = prior_make(priors,'offset',unif,Mean,width,rmin,rmax, strt)
                    unif,Mean,width,rmin,rmax, strt = prior_make(priors,'offset',unif,Mean,width,rmin,rmax, strt)

        
    if "T" in comps: 
        
        if param_info["tpo"] == "TPO_01":
            unif,Mean,width,rmin,rmax, strt = prior_make(priors,'pcs',unif,Mean,width,rmin,rmax, strt)
            unif,Mean,width,rmin,rmax, strt = prior_make(priors,'ks',unif,Mean,width,rmin,rmax, strt)
            unif,Mean,width,rmin,rmax, strt = prior_make(priors,'n',unif,Mean,width,rmin,rmax, strt)
        if param_info["tpo"] == "TPO_02":
            unif,Mean,width,rmin,rmax, strt = prior_make(priors,'pcb',unif,Mean,width,rmin,rmax, strt)
            unif,Mean,width,rmin,rmax, strt = prior_make(priors,'k0',unif,Mean,width,rmin,rmax, strt)
            unif,Mean,width,rmin,rmax, strt = prior_make(priors,'n',unif,Mean,width,rmin,rmax, strt)
        if param_info["tpo"] == "TPO_03":
            unif,Mean,width,rmin,rmax, strt = prior_make(priors,'ke',unif,Mean,width,rmin,rmax, strt)
            unif,Mean,width,rmin,rmax, strt = prior_make(priors,'yt',unif,Mean,width,rmin,rmax, strt)

        unif,Mean,width,rmin,rmax , strt= prior_make(priors,'beta',unif,Mean,width,rmin,rmax, strt)
            
    print(len(Mean),'parameters to be estimated')
    return  unif, Mean, width, rmin, rmax, strt

def prior_make(priors, param, unif, Mean, width, rmin, rmax, strt): 
    unif.append(priors[param]['unif'])

    if priors[param]['unif'] == True:
        Mean.append(priors[param]['min']+ ((priors[param]['max']-priors[param]['min'])/2))
        width.append((priors[param]['max']-priors[param]['min'])/2) 
    else:
        Mean.append(priors[param]['mea'] )
        width.append(priors[param]['ste'] ) 
        
    rmin.append(priors[param]['min'])
    rmax.append(priors[param]['max'])
    strt.append(priors[param]['strt'])
    
    return unif, Mean, width, rmin, rmax, strt
