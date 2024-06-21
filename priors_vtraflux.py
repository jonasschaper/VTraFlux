# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:13:33 2024

@author: Jonas Schaper
"""



####################################################################################################################################################################################### 
#######################################################################################################################################################################################

def prior_range_get(prange):
    

    if prange == "erpe":
    
        priors = { 'vv':    {'min': 0.0025, 'max': 0.15, 'uniform': True,  'log': False },
                   'al':    {'min': 0.005,  'max': 0.1, 'uniform': True,  'log': False },
                   'shift': {'min': -0.1,   'max': 0.1,  'uniform': True,  'log': False   }
                   }   
        
        
    print('prior range: ', prange)
    return priors


#######################################################################################################################################################################################
#######################################################################################################################################################################################


def load_priors(ndays, comps,param_info, prange):
    
    
    priors = prior_range_get(prange)
    
    Log   = [];   uniform  = [];     Mean  = []
    width = [];   rmin  = [];     rmax  = []
    # uniformrom conservative parameter priors 
    
    if "EC" in comps: 
        Log,uniform,Mean,width,rmin,rmax = prior_make(priors,'al',Log,uniform,Mean,width,rmin,rmax)    
 
    if param_info["v_type"] == "variable":
        ittt = 1
        if param_info["st_t"] == 0: ittt = 0
        for i in range(int( (ndays*24/param_info["v_int"])+1 - ((param_info["st_t"]/param_info["v_int"])-ittt) )): 
                    Log,uniform,Mean,width,rmin,rmax = prior_make(priors,'vv',Log,uniform,Mean,width,rmin,rmax)
    else:
        Log,uniform,Mean,width,rmin,rmax = prior_make(priors,'vv',Log,uniform,Mean,width,rmin,rmax)

    if "EC" in comps and param_info["offset"] == True:     
        if param_info["o_type"] == "constant":
            Log,uniform,Mean,width,rmin,rmax = prior_make(priors,'shift',Log,uniform,Mean,width,rmin,rmax)
        if param_info["o_type"] == "variable":
            for i in range(int(ndays*24/param_info["o_int"]+1)): 
                    Log,uniform,Mean,width,rmin,rmax = prior_make(priors,'shift',Log,uniform,Mean,width,rmin,rmax)
        if param_info["o_type"] == "linear":
                    Log,uniform,Mean,width,rmin,rmax = prior_make(priors,'shift',Log,uniform,Mean,width,rmin,rmax)
                    Log,uniform,Mean,width,rmin,rmax = prior_make(priors,'shift',Log,uniform,Mean,width,rmin,rmax)

        
    if "T" in comps: 
        Log,uniform,Mean,width,rmin,rmax = prior_make(priors,'beta',Log,uniform,Mean,width,rmin,rmax)
        Log,uniform,Mean,width,rmin,rmax = prior_make(priors,'pcs',Log,uniform,Mean,width,rmin,rmax)
        Log,uniform,Mean,width,rmin,rmax = prior_make(priors,'ka',Log,uniform,Mean,width,rmin,rmax)
        Log,uniform,Mean,width,rmin,rmax = prior_make(priors,'n',Log,uniform,Mean,width,rmin,rmax)
    
    print(len(Mean),'parameters to be estimated')
    return Log, uniform, Mean, width, rmin, rmax

def prior_make(priors, param,Log, uniform, Mean, width, rmin, rmax): 
    Log.append(priors[param]['log'] )
    uniform.append(priors[param]['uniform'])

    if priors[param]['uniform'] == True:
        Mean.append(priors[param]['min']+ ((priors[param]['max']-priors[param]['min'])/2))
        width.append((priors[param]['max']-priors[param]['min'])/2) 
    else:
        Mean.append(priors[param]['mea'] )
        width.append(priors[param]['ste'] ) 
        
    rmin.append(priors[param]['min'])
    rmax.append(priors[param]['max'])   
    return Log, uniform, Mean, width, rmin, rmax
