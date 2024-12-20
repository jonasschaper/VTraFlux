# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:32:30 2018

@author: jonas schaper

"""
import os
import tempfile
import importlib.util
from scipy.sparse import diags
from scipy import interpolate
import random as rand


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from vtraflux import *


global ini_cons
ini_cons = False

#%% # starting the MCMC
if __name__ == '__main__':
    
    method = sim_type["method"] # e.g., 'leastsq' #'nelder'
    #%% # starting the joint - L curve method 

    if lcurve_est == 'joint':

        script = 'lmfit_vtraflux.py' 
        with open(dname + '/addray_vtraflux.py') as file:
            script_content = file.read()
        exec(script_content) 
    

        sites      = dat_info["sites"]
        for site in sites:
          depthpairs      = dat_info["depths_T"]
          for depthpair in depthpairs:
            compounds      = dat_info["compounds"]
            for comps in compounds: 
              for h_factor in par_info['v_ints']:
                    
                    par_info['st_t'] = 0
                    par_info['v_int'] = h_factor
                    weights      = run_info["weights"]
                    
                    #----------------------------------------------------------------
                    postrun = run_info["postrun"]
                    postsim = run_info["postsim"]
    
                    Ds,  dirName2s, indata, result_df, npars, seclocs , dirName_run = DREAM_ecmod_create(site, comps,  dat_info, par_info, run_info); R_hat = []
                    print(Ds)
                    if "EC" in comps and "T" in comps:
                        seclocs_EC = seclocs[0]; seclocs_T = seclocs[1]
                        indata_EC = indata[0]; indata_T = indata[1]            
                    elif "EC" in comps:
                        seclocs_EC = seclocs
                        indata_EC = indata
                    elif "T" in comps:
                        seclocs_T = seclocs
                        indata_T = indata
            

                    if run_info["vantage"]: 
                        for nw in range(len(weights)):
                                Ds[nw].chains[0].current=  np.array(result_df[nw])[0][1:npars+1]
    
                    print(  'iteration Start= ',time.strftime("%Y-%m-%d %H:%M:%S"));  start = time.time()
                    
                    nprocs_avail = psutil.cpu_count(logical=False)
                    print( 'psutil num_cpus= ',nprocs_avail)  
                    if run_info["supercomputer"]:
                        nprocs_avail = len(os.sched_getaffinity(0))  
                        print( 'supercomputer num_cpus= ',nprocs_avail) 
    
    
                    nprocs = len(weights)
                    print( 'num_cpus used = number of weights =  ', nprocs)  
                    
                    
                    if nprocs_avail < len(weights): warnings.warn("Number of weights larger than number of processors.")
                
                    ray.init(num_cpus=nprocs, ignore_reinit_error=True)
                    assert ray.is_initialized() == True
                    
                    likered  = 1
                    if "EC" in comps: 
                        if "T" in comps:
                            rbc  = np.ones(len(indata_EC[1][1]))*   np.mean(indata_EC[1][2]) # groundwater EC value
                        else:
                            rbc = 0.0; 
                        ec_model = vtraflux_model(indata_EC,dat_info["chunk"],dt = 0.05, rbc = rbc,  seclocs= seclocs_EC, param_info = par_info,comp="EC", ID = 0)# for i in range(len(weights))
    
                    if "T" in comps:
                        if len(seclocs_T) > 2: 
                            rbc = indata_T[2][2]
                            if "EC" in comps:
                                #likered = len(indata_T[0][1])/len(indata_EC[0][1])
                                T_model=vtraflux_model(indata_T,dat_info["chunk"],dt = 0.025, rbc = rbc,   seclocs= seclocs_T, param_info = par_info,comp="EC_T",ID = 0)# for i in range(len(weights))]
                            else:
                                T_model=vtraflux_model(indata_T,dat_info["chunk"],dt = 0.025, rbc = rbc,   seclocs= seclocs_T, param_info = par_info,comp="T",ID =0) #for i in range(len(weights))]
    
                    idx = 0
                    comps_dict = {'comps': comps}
                    
                    models_dict_l = []
                    for i in range(int(len(weights))):            
                        if comps == "EC_T":
                            models =[ {'T': T_model, 'EC': ec_model, 'wv':weights[i],  'wo': weights[i],   'dirName' : dirName2s[i] } ]
                        if comps == "EC":
                            models =[ { 'EC': ec_model, 'wv': weights[i],  'wo': weights[i],  'dirName' : dirName2s[i] } ]
                        if comps == "T":
                            models =[ {'T': T_model, 'wv':weights[i],  'wo': weights[i],  'dirName' : dirName2s[i] } ]
                        models_dict_l.append(models)
                    
                    vtralm_models = [modified_script.vtraflux_lmfit.remote(dream = Ds[i], method = method,  models = models_dict_l[i],  comps = comps_dict, ID = i) for i in range(len(weights))]
    
                    result_ids = []
                    for o in  range (nprocs):
                        #idx = int(i*nprocs+o)
                        idx = int(o)
                        result_ids.append(vtralm_models[o].model_run.remote())
                    results = ray.get(result_ids)
                    
                    print( 'lmfit has finished estimating parameters') # 
                    
                    result_ids = []
                    for o in  range (nprocs):
                        #idx = int(i*nprocs+o)
                        idx = int(o)
                        result_ids.append(vtralm_models[o].print_results.remote())
                    results = ray.get(result_ids)
                    
                    
                    print( 'lmfit has finished printing results') # 
                    ray.shutdown()
                    assert ray.is_initialized() == False    
    
    
        #from regular_vtraflux import *
        #find_weight(directory = dirName_run, erfac =1,  postest = True,  mco = "MCO_02", printres = True)
    
    
    #%% # lcurve_est via single processor runs  

    if lcurve_est == 'single':
        
        from lmfit_vtraflux import *

        sites      = dat_info["sites"]
        for site in sites:
          depthpairs      = dat_info["depths_T"]
          for depthpair in depthpairs:
            compounds      = dat_info["compounds"]
            for comps in compounds: 
              for h_factor in par_info['v_ints']:
                    
                    par_info['st_t']  = 0
                    par_info['v_int'] = h_factor
                    weights           = run_info["weights"]
                    
                    Ds,  dirName2s, indata, result_df, npars, seclocs , dirName_run = DREAM_ecmod_create(site, comps,  dat_info, par_info, run_info); R_hat = []
                       
                    for i in range(len(weights)):
                        w = weights[i]
                        wo = w; wv = w; 

                        postrun = run_info["postrun"]
                        postsim = run_info["postsim"]

                        if postrun: vantage_w = w
                        
                        dirName = dirName2s[i]
                        D       = Ds[i]
                        
                        print(dirName)
                        if "EC" in comps and "T" in comps:
                            seclocs_EC = seclocs[0]; seclocs_T = seclocs[1]
                            indata_EC = indata[0]; indata_T = indata[1]            
                        elif "EC" in comps:
                            seclocs_EC = seclocs
                            indata_EC = indata
                        elif "T" in comps:
                            seclocs_T = seclocs
                            indata_T = indata
                
                        if run_info["vantage"]: 
                            for i in range(D.nc): D.chains[i].current=  np.array(result_df)[0][i][1:npars+1]


                        print(  'iteration Start= ',time.strftime("%Y-%m-%d %H:%M:%S"));  start = time.time()                    

                        likered  = 1
                        
                        if "EC" in comps: 
                            if "T" in comps:
                                rbc  = np.ones(len(indata_EC[1][1]))*   np.mean(indata_EC[1][2]) # groundwater EC value
                            else:
                                rbc = 0.0; 
                            ec_model = vtraflux_model(indata_EC,dat_info["chunk"],dt = 0.05, rbc = rbc,  seclocs= seclocs_EC, param_info = par_info,comp="EC", ID = 0)# for i in range(len(weights))
                        if "T" in comps:
                            if len(seclocs_T) > 2: 
                                rbc = indata_T[2][2]
                                if "EC" in comps:
                                    T_model=vtraflux_model(indata_T,dat_info["chunk"],dt = 0.025, rbc = rbc,   seclocs= seclocs_T, param_info = par_info,comp="EC_T",ID = 0)# for i in range(len(weights))]
                                else:
                                    T_model=vtraflux_model(indata_T,dat_info["chunk"],dt = 0.025, rbc = rbc,   seclocs= seclocs_T, param_info = par_info,comp="T",ID =0) #for i in range(len(weights))]
        
                        idx = 0
                        comps_dict = {'comps': comps}
                        models_dict_l = []
                        if comps == "EC_T":
                            models =[ {'T': T_model, 'EC': ec_model, 'wv':w,  'wo': w, 'dirName' : dirName } ]
                        if comps == "EC":
                            models =[ { 'EC': ec_model, 'wv':w,  'wo': w, 'dirName' : dirName } ]
                        if comps == "T":
                            models =[ {'T': T_model, 'wv':w,  'wo': w, 'dirName' : dirName } ]
                        models_dict_l.append(models)
                        
                        vtralm = vtraflux_lmfit(dream = D, method = method,  models = models_dict_l[0],  comps = comps_dict, ID = None)
                        mini_res = vtralm.model_run()
                        print( 'lmfit has finished with weight:', w) # 
                        vtralm.print_results()
                        
                    #%% # Postprocessing: determining optimal L curve weight     
                    #from regular_vtraflux import *
                    #find_weight(directory = dirName_run, erfac =1,  postest = True,  mco = "MCO_02", printres = True)              