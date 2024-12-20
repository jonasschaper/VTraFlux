# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:32:30 2018

@author: jonas schaper

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



#%% # starting the MCMC

global ini_cons
ini_cons = False


if __name__ == '__main__':
    
    from vtraflux import *

    if lcurve_est == 'single':
        print('DREAM run started in single-Lcurve est. mode-------------------------------')
        print('looping through,',str(len(run_info["weights"])), ' weighting factors -------------------------------')

        script = 'vtraflux.py' 
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
                    
                    #par_info['st_t'] = 0
                    par_info['v_int'] = h_factor
                    weights      = run_info["weights"]
                       
                    for w in weights:
                        wo = w; wv = w; 
    
                        postrun = run_info["postrun"]
                        postsim = run_info["postsim"]
    
                        if postrun: vantage_w = w
                        
                        
                        Ds,  dirName2s, indata, result_df, npars, seclocs , dirName = DREAM_ecmod_create(site, comps,  dat_info, par_info, run_info); R_hat = []
                        
                        dirName = dirName2s[weights.index(w)]
                        D       = Ds[weights.index(w)]

                        
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
                        
                        '''
                        nprocs = psutil.cpu_count(logical=False)
                        print( 'psutil num_cpus= ',nprocs) # very important to match number of chains to number of cores 
                        if run_info["supercomputer"]:
                            nprocs = len(os.sched_getaffinity(0))  
                            print( 'supercomputer num_cpus= ',nprocs) # very important to match number of chains to number of cores 
                        if run_info["workstation"]:
                            if h_factor != 3 or h_factor != 2:
                                nprocs = nprocs-2
                            if comps == "EC_T":
                                nprocs = 14; D.nc = 14
                                nprocs = 12; D.nc = 24                    
                        '''
                        
                        print( 'nchains= ',D.nc) #  match number of chains to number of cores 
                        print( 'num_cpus used= ',nprocs) 
                    

                        print('multicore ray')
                        ray.init(num_cpus=nprocs, ignore_reinit_error=True)
                        assert ray.is_initialized() == True
                        likered  = 1
                        
                        if "EC" in comps: 
                            if "T" in comps:
                                rbc  = np.ones(len(indata_EC[1][1]))*   np.mean(indata_EC[1][2]) # groundwater EC value
                            else:
                                rbc = 0.0; 
                            ec_models = [modified_script.vtraflux_model.remote(indata_EC,dat_info["chunk"],dt = 0.05, rbc = rbc,  seclocs= seclocs_EC, param_info = par_info,comp="EC", ID = i) for i in range(D.nc)]
                        
                        
                        if "T" in comps:
                            if len(seclocs_T) > 2: 
                                rbc = indata_T[2][2]
                                if "EC" in comps:
                                    T_models = [modified_script.vtraflux_model.remote(indata_T,dat_info["chunk"],dt = 0.025, rbc = rbc,   seclocs= seclocs_T, param_info = par_info,comp="EC_T",ID = i) for i in range(D.nc)]
                                else:
                                    T_models = [modified_script.vtraflux_model.remote(indata_T,dat_info["chunk"],dt = 0.025, rbc = rbc,   seclocs= seclocs_T, param_info = par_info,comp="T",ID = i) for i in range(D.nc)]
    
    
                        if "EC" in comps:  
                            for i in range(int(D.nc/nprocs)):            
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(ec_models[idx].update_params.remote(D.chains[idx].current, comps,  ini_cons = ini_cons))
                
                                results = ray.get(result_ids)
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(ec_models[idx].simulation.remote())
                                results = ray.get(result_ids)
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(ec_models[idx].loglikelihood.remote(D.chains[idx],dirName, start = True))
                                results_ec = ray.get(result_ids)     
    
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    D.chains[idx].Lold    =  results_ec[o][0]*likered + results_ec[o][4] * wv + results_ec[o][5]* wo             
                                    D.chains[idx].rmse_old= results_ec[o][2]; 
                                    D.chains[idx].sse_old = results_ec[o][3]
                                    D.chains[idx].fitlike_old = results_ec[o][0]
    
                        if "T" in comps:  
                            for i in range(int(D.nc/nprocs)):            
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(T_models[idx].update_params.remote(D.chains[idx].current, comps,  ini_cons = ini_cons))
                                results2 = ray.get(result_ids); del results2
                                
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(T_models[idx].simulation.remote())
                                results = ray.get(result_ids); del results
                                
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(T_models[idx].loglikelihood.remote(D.chains[idx], dirName, start = True))
                                results = ray.get(result_ids)     
                                #print(results)
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    if "EC" in comps and "T" in comps:
                                        D.chains[idx].Lold     +=  results[o][0]    
                                        D.chains[idx].rmse_old += results[o][2];  
                                        D.chains[idx].sse_old += results[o][3]   
                                        D.chains[idx].fitlike_old += results[o][0]
                                        
                                    else:
                                        D.chains[idx].Lold    =  results[o][0] + results[o][4] * wv + results[o][5]* wo             
                                        D.chains[idx].rmse_old= results[o][2]; 
                                        D.chains[idx].sse_old = results[o][3]
                                        D.chains[idx].fitlike_old = results[o][0]
    
                        print("duration =", time.time() - start)
                    
                    #%% #Burn in start
                        print('multicore ray burn in phase')
                        while D.burn == True:   
                            print(  'iteration start= ',time.strftime("%Y-%m-%d %H:%M:%S")); start = time.time()
                        
                            D.std_cal(); #print(D.Xstd) # compute stds for each parameter for pCR adaption 
                            D.propgen() # obtaining a new set of proposed parameter values 
                            
                            if "EC" in comps:  
                                for i in range(int(D.nc/nprocs)):
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(ec_models[idx].update_params.remote(D.chains[idx].proposal, comps,  ini_cons = ini_cons))
                                    results = ray.get(result_ids)
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(ec_models[idx].simulation.remote())
                                    results = ray.get(result_ids)
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(ec_models[idx].loglikelihood.remote(D.chains[idx],dirName, start = False))
                                    results_ec = ray.get(result_ids)     
                                            
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        D.chains[idx].Lnew    =  results_ec[o][0]*likered + results_ec[o][4] * wv + results_ec[o][5]* wo  
                                        D.chains[idx].rmse_new = results_ec[o][2]; D.chains[idx].sse_new = results_ec[o][3]
                                        D.chains[idx].difv_new =  results_ec[o][4]
                                        D.chains[idx].difo_new =  results_ec[o][5]; 
                                        D.chains[idx].fitlike_new = results_ec[o][0]
    
                                
                            if "T" in comps:  
                                for i in range(int(D.nc/nprocs)):
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(T_models[idx].update_params.remote(D.chains[idx].proposal, comps,  ini_cons = ini_cons))
                                    results = ray.get(result_ids)
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(T_models[idx].simulation.remote())
                                    results = ray.get(result_ids)
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(T_models[idx].loglikelihood.remote(D.chains[idx],dirName, start = False))
                                    results = ray.get(result_ids)     
                                            
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        if "EC" in comps and "T" in comps:
                                            D.chains[idx].Lnew    +=  results[o][0] #+ results[o][4] * wv + results[o][5]* wo             
                                            D.chains[idx].rmse_new += results[o][2]; 
                                            D.chains[idx].sse_new += results[o][3]     
                                            D.chains[idx].fitlike_new += results[o][0]
    
    
                                        else:
                                            D.chains[idx].Lnew    =  results[o][0] + results[o][4] * wv + results[o][5]* wo  
                                            
                                            D.chains[idx].rmse_new= results[o][2]; D.chains[idx].sse_new = results[o][3]
                                            D.chains[idx].difv_new =  results[o][4]
                                            D.chains[idx].difo_new =  results[o][5]; 
                                            D.chains[idx].fitlike_new = results[o][0]
                        
                            for i in range(D.nc):     
                                D.chains[i].tprob() 
                                D.J[D.chains[i].idd]    += np.nansum ( ( D.chains[i].jump / D.Xstd )**2  )
                                D.n_id[D.chains[i].idd] += 1   
                                D.nacc   += D.chains[i].accept 
                    
                            D.update_R_hats(dirName, R_hat)
                            
                            if D.ct != 0 and D.ct >= 5: 
                                D.Chain_removal( D.tc); # omega calculated using the last half of all likelihoods   
                                D.pCR_update();   # print(D.pCR)   # update selection probabilities of 
                                # smaller idd vlaues mean smaller threshold CR values and thus a larger subset of samples (less pars to be kept)
                                
                            if  D.ct != 0 and D.ct >= 5:  print('acceptance rate', D.nacc /((D.ct)* D.nc) )  
                        
                            D.print_recovery(dirName)
                            D.update_current(dirName) # updates current.dat file with current state of chains
                            D.update_pars(dirName)

                            gc.collect()
    
                            if max(D.R) < 1.2 and D.ct > int(D.rac/100*D.nc): 
                                D.burn = False 
                            elif max(D.R) < 1.2 and D.tc > int(250): 
                                D.burn = False

                            if postrun: 
                                  if D.ct == 100: D.burn = False
                            print("duration =", time.time() - start)
                        
                        print("burn in finished after", D.ct, "iterations")
                        D.nb = D.ct  # keeps track on how many iterations were needed to achieve convergence
                        # End of BURN-IN -----------------------------------------------------           
                        
                        #%% # Sampling of posterior starts ---------------------------------------------   
                        D.ct = 0; D.nacc = 0
    
                        for i in range(D.nc):
                            D.chains[i].postpars     = np.append(np.append(np.append(np.append(np.append( np.append(np.copy(D.chains[i].current), np.copy(D.chains[i].rmse_new)), np.copy(D.chains[i].sse_new)), np.copy(D.chains[i].fitlike_new)), np.copy(D.chains[i].difo_new)), np.copy(D.chains[i].difv_new)),0)    # current parameters will be stored in pars
                            D.chains[i].postlike     =  [D.chains[i].Lold]
                        
                        print('multicore ray posterior sampling phase')
                        while D.ct < D.rac:
                            D.propgen()                        
                            if "EC" in comps:  
                                for i in range(int(D.nc/nprocs)):
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(ec_models[idx].update_params.remote(D.chains[idx].proposal, comps,  ini_cons = ini_cons))
                                    results = ray.get(result_ids)
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(ec_models[idx].simulation.remote())
                                    results = ray.get(result_ids)
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(ec_models[idx].loglikelihood.remote(D.chains[idx],dirName, start = False))
                                    results_ec = ray.get(result_ids)     
                                    for o in range (nprocs):
                                        idx = int(i*nprocs+o)
                                        D.chains[idx].Lnew    =  results_ec[o][0]*likered + results_ec[o][4] * wv + results_ec[o][5]* wo             
                                        D.chains[idx].rmse_new= results_ec[o][2]; D.chains[idx].sse_new = results_ec[o][3]
                                        D.chains[idx].difv_new =  results_ec[o][4]
                                        D.chains[idx].difo_new =  results_ec[o][5]; 
                                        D.chains[idx].fitlike_new = results_ec[o][0]
      
    
                            if "T" in comps:  
                                for i in range(int(D.nc/nprocs)):
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(T_models[idx].update_params.remote(D.chains[idx].proposal, comps,  ini_cons = ini_cons))
                                    results = ray.get(result_ids)
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(T_models[idx].simulation.remote())
                                    results = ray.get(result_ids)
                                    
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(T_models[idx].loglikelihood.remote(D.chains[idx],dirName, start = False))
                                    results = ray.get(result_ids)    
                                                            
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        if "EC" in comps and "T" in comps:
                                            D.chains[idx].Lnew    +=  results[o][0]              
                                            D.chains[idx].rmse_new += results[o][2]; 
                                            D.chains[idx].sse_new += results[o][3]   
                                            D.chains[idx].fitlike_new += results[o][0]
    
                                        else:
                                            D.chains[idx].Lnew    =  results[o][0] + results[o][4] * wv + results[o][5]* wo             
                                            D.chains[idx].rmse_new= results[o][2]; D.chains[idx].sse_new = results[o][3]
                                            D.chains[idx].difv_new =  results[o][4]
                                            D.chains[idx].difo_new =  results[o][5];  
                                            D.chains[idx].fitlike_new = results[o][0]
    
                                gc.collect()
                                                    
                            for i in range(int(D.nc/nprocs)):
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    D.chains[idx].tprob()        
                                if postsim:
                                    if "T" in comps:  
                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            if D.chains[idx].accept == 1:
                                                result_ids.append(T_models[idx].v_o_int.remote())
                                            else:
                                                result_ids.append(T_models[idx].dummy_fun.remote())
                                        results_vv = ray.get(result_ids)
    
        
                                    if "EC" in comps:  
                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            if D.chains[idx].accept == 1:
                                                result_ids.append(ec_models[idx].tau_cal.remote())
                                            else:
                                                result_ids.append(ec_models[idx].dummy_fun.remote())
                                        tau_results_ec = ray.get(result_ids)
            
                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            if D.chains[idx].accept == 1:
                                                result_ids.append(ec_models[idx].v_o_int.remote())
                                            else:
                                                result_ids.append(ec_models[idx].dummy_fun.remote())
                                        results_vv_ec = ray.get(result_ids)
    
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        if D.chains[idx].accept == 1 and D.nacc == 0:
                                            if "T" in comps:  
                                                Tcmoddf = pd.DataFrame(results[o][9])
                                                Tcmoddf['cmea'] = results[o][7]
                                                Tcmoddf['c_in'] =results[o][8]
                                                Tcmoddf['c_ou'] =results[o][10]
                                                
                                                Tvdf = pd.DataFrame(results_vv[o][0])
                                                Tddf = pd.DataFrame(results_vv[o][0])
    
                                            if "EC" in comps:  
                                                ecmoddf = pd.DataFrame(results_ec[o][9])
                                                ecmoddf['cmea'] = results_ec[o][7]
                                                ecmoddf['c_in'] =results_ec[o][8]
                                                evdf = pd.DataFrame(results_vv_ec[o][0])
                                                odf = pd.DataFrame(results_vv_ec[o][0])   
                                                etaudf = pd.DataFrame(tau_results_ec[o][1])
                                                
                                        if D.chains[idx].accept == 1:                        
                                            D.nacc   += D.chains[idx].accept 
                                            if "T" in comps:  
                                                Tvdf['sim' + str(D.nacc)] = results_vv[o][1]
                                                Tddf['sim' + str(D.nacc)] = results_vv[o][2]
                                                Tcmoddf['sim' + str( D.nacc)] = results[o][1]
    
                                            if "EC" in comps:  
                                                evdf['sim' + str(D.nacc)] = results_vv_ec[o][1]
                                                odf['sim' + str(D.nacc)] = results_vv_ec[o][2]
                                                etaudf['sim' + str(D.nacc)] = tau_results_ec[o][0]
                                                ecmoddf['sim' + str( D.nacc)] = results_ec[o][1]
            
                            if D.ct > 10 and D.ct%5  == 0: print(D.ct,D.R)
                            if D.ct != 0 and D.ct%20 == 0: print('acceptance rate', D.nacc /((D.ct)* D.nc ))
                        
                            D.update_current( dirName) 
                            D.update_R_hats(dirName, R_hat)
                            D.update_pars(dirName)

                        D.print_postpars(dirName)
                        
                        #%% # Saving posteriors and simulated state vars ---------------------------------------------   
    
                        if postsim:
    
                            foldname4  = dirName
                            rst      = np.loadtxt(foldname4+'/postpars.dat' ) # 
                            rstpd = pd.DataFrame(rst)
                            result_df = rstpd 
                            
                            bestres = np.array(result_df.nlargest(n=1, columns=[0]))
                            bestparam = bestres[0][1:npars+1]
        
                            if "T" in comps:  
                                Tcmoddf.to_csv(os.path.join(foldname4) +  '/'+ "T_cmod_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')
                                Tvdf.to_csv(os.path.join(foldname4) +  '/'+ "T_v_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.5f')
                                Tddf.to_csv(os.path.join(foldname4) +  '/'+ "T_d_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.6f')
                    
                                T_models[0].update_params.remote(bestparam, comps,   ini_cons = ini_cons)
                                T_models[0].simulation.remote()
                                like, cmod, rmse, sse, vvdlike, oodlike, parlike, cmea, lbc, modtime ,rbcT   = ray.get( T_models[0].loglikelihood.remote(D.chains[0],dirName, start = False))
                                #tau_vec, timev,tau_vec2 = ray.get( T_models[0].tau_cal.remote())
                                modt, vvs, oos = ray.get( T_models[0].v_o_int.remote())
                                
                                f = open(os.path.join(foldname4) +  '/'+ "T_cmod_ml" + '.txt','w')
                                dim = np.shape(cmod)
                                for k in range(dim[0]):
                                    f.write('%g %g %g \n' %  (modtime[k], cmea[k] , cmod[k]) )        
                                f.write('\n')
                                f.close()
                                
                                f = open(os.path.join(foldname4) +  '/'+ "T_v_ml" + '.txt','w')
                                dim = np.shape(modt)
                                for k in range(dim[0]):
                                    f.write('%g %g \n' %  (modt[k], vvs[k]) )        
                                f.write('\n')
                                f.close()
    
                                f = open(os.path.join(foldname4) +  '/'+ "T_d_ml" + '.txt','w')
                                dim = np.shape(modt)
                                for k in range(dim[0]):
                                    f.write('%g %s \n' %  (modt[k],   str('{:04.5f}'.format(oos[k]) )     ) )        
                                f.write('\n')
                                f.close()
    
        
                            if "EC" in comps:  
                                
                                etaudf.to_csv(os.path.join(foldname4) +  '/'+ "ec_tau_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')
                                ecmoddf.to_csv(os.path.join(foldname4) +  '/'+ "ec_cmod_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')
                                evdf.to_csv(os.path.join(foldname4) +  '/'+ "ec_v_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')
                                odf.to_csv(os.path.join(foldname4) +  '/'+ "oo_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')
                                
                                ec_models[0].update_params.remote(bestparam, comps,   ini_cons = ini_cons)
                                ec_models[0].simulation.remote()
                                like, cmod, rmse, sse, vvdlike, oodlike, parlike, cmea, lbc, modtime, rbcEC    = ray.get( ec_models[0].loglikelihood.remote(D.chains[0],dirName, start = False))
                                tau_vec, timev, tau_vec2 = ray.get( ec_models[0].tau_cal.remote())
                                modt, vvs, oos = ray.get( ec_models[0].v_o_int.remote())
                                
                    
                                f = open(os.path.join(foldname4) +  '/'+ "ec_cmod_ml" + '.txt','w')
                                dim = np.shape(cmod)
                                for k in range(dim[0]):
                                    f.write('%g %g %g \n' %  (modtime[k], cmea[k] , cmod[k]) )        
                                f.write('\n')
                                f.close()
                                
                                f = open(os.path.join(foldname4) +  '/'+ "ec_v_ml" + '.txt','w')
                                dim = np.shape(modt)
                                for k in range(dim[0]):
                                    f.write('%g %g \n' %  (modt[k], vvs[k]) )        
                                f.write('\n')
                                f.close()
                                
                                f = open(os.path.join(foldname4) +  '/'+ "o_ml" + '.txt','w')
                                dim = np.shape(modt)
                                for k in range(dim[0]):
                                    f.write('%g %g \n' %  (modt[k], oos[k]) )        
                                f.write('\n')
                                f.close()                                
                                
                                f = open(os.path.join(foldname4) +  '/'+ "ec_tau_ml" + '.txt','w')
                                dim = np.shape(tau_vec)
                                for k in range(dim[0]):
                                    f.write('%g %g \n' %  (timev[k], tau_vec[k]) )        
                                f.write('\n')
                                f.close()
                                
                        ray.shutdown()
                        assert ray.is_initialized() == False    
                        

                        # End of DREAM -----------------------------------------------------  
                        D.print_Rhats(dirName, R_hat)
                        D.print_DREAM_settings(dirName)


#%%


    if lcurve_est == 'joint':
        
        print('DREAM run started in joint-Lcurve est. mode-------------------------------')
        
        script = 'vtraflux.py' 

        with open(dname + '/addray_vtraflux.py') as file:
            script_content = file.read()
        exec(script_content) 

        
        sites      = dat_info["sites"]
        ray.init(num_cpus=nprocs, ignore_reinit_error=True)
        assert ray.is_initialized() == True


        for site in sites:
          depthpairs      = dat_info["depths_T"]
          for depthpair in depthpairs:
            compounds      = dat_info["compounds"]
            for comps in compounds: 
              for h_factor in par_info['v_ints']:
                    
                    par_info['v_int'] = h_factor
                    weights      = run_info["weights"]
                       
                    postrun = run_info["postrun"]
                    postsim = run_info["postsim"]

                   
                    Ds,  dirNames, indata, result_df, npars, seclocs, dirName_run = DREAM_ecmod_create(site, comps, dat_info, par_info, run_info); R_hat = []

            
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
                            for i in range(Ds[nw].nc): 
                                Ds[nw].chains[i].current=  np.array(result_df[nw])[i][1:npars+1]

                    
                    
                    
                    print(  'iteration Start= ',time.strftime("%Y-%m-%d %H:%M:%S"));  start = time.time()
                    print( 'num_cpus used= ',nprocs)  
                    likered  = 1 

                    for nw in range(len(weights)):
                        if "EC" in comps: 
                            if "T" in comps:
                                rbc  = np.ones(len(indata_EC[1][1]))*   np.mean(indata_EC[1][2]) # groundwater EC value
                            else:
                                rbc = 0.0; 
                            ec_models = [modified_script.vtraflux_model.remote(indata_EC,dat_info["chunk"],dt = 0.05, rbc = rbc,  seclocs= seclocs_EC, param_info = par_info,comp="EC", ID = i) for i in range(Ds[nw].nc)]
                        
                        if "T" in comps:
                            if len(seclocs_T) > 2: 
                                rbc = indata_T[2][2]
                                #rbc = 0.0; 
                                if "EC" in comps:
                                    T_models = [modified_script.vtraflux_model.remote(indata_T,dat_info["chunk"],dt = 0.025, rbc = rbc,   seclocs= seclocs_T, param_info = par_info,comp="EC_T",ID = i) for i in range(Ds[nw].nc)]
                                else:
                                    T_models = [modified_script.vtraflux_model.remote(indata_T,dat_info["chunk"],dt = 0.025, rbc = rbc,   seclocs= seclocs_T, param_info = par_info,comp="T",ID = i) for i in range(Ds[nw].nc)]
                                    
                    print('multicore ray')
                    for nw in range(len(weights)):
                        wo = weights[nw]; wv = weights[nw]; 
                        
                        if "EC" in comps:  
                            for i in range(int(Ds[nw].nc/nprocs)):            
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(ec_models[idx].update_params.remote(Ds[nw].chains[idx].current, comps,  ini_cons = ini_cons))
                
                                results = ray.get(result_ids)
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(ec_models[idx].simulation.remote())
                                results = ray.get(result_ids)
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(ec_models[idx].loglikelihood.remote(Ds[nw].chains[idx],dirNames[nw], start = True))
                                results_ec = ray.get(result_ids)     

                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    Ds[nw].chains[idx].Lold    =  results_ec[o][0]*likered + results_ec[o][4] * wv + results_ec[o][5]* wo             
                                    Ds[nw].chains[idx].rmse_old= results_ec[o][2]; 
                                    Ds[nw].chains[idx].sse_old = results_ec[o][3]
                                    Ds[nw].chains[idx].fitlike_old = results_ec[o][0]

                            #del results_ec  
                        if "T" in comps:  
                            for i in range(int(Ds[nw].nc/nprocs)):            
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(T_models[idx].update_params.remote(Ds[nw].chains[idx].current, comps,  ini_cons = ini_cons))
                                results2 = ray.get(result_ids); del results2
                                
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(T_models[idx].simulation.remote())
                                results = ray.get(result_ids); del results
                                
                                result_ids = []
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    result_ids.append(T_models[idx].loglikelihood.remote(Ds[nw].chains[idx], dirNames[nw], start = True))
                                results = ray.get(result_ids)     
                                #print(results)
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    if "EC" in comps and "T" in comps:
                                        Ds[nw].chains[idx].Lold     +=  results[o][0]    
                                        Ds[nw].chains[idx].rmse_old += results[o][2];  
                                        Ds[nw].chains[idx].sse_old += results[o][3]   
                                        Ds[nw].chains[idx].fitlike_old += results[o][0]
                                        
                                    else:
                                        Ds[nw].chains[idx].Lold    =  results[o][0] + results[o][4] * wv + results[o][5]* wo             
                                        Ds[nw].chains[idx].rmse_old= results[o][2]; 
                                        Ds[nw].chains[idx].sse_old = results[o][3]
                                        Ds[nw].chains[idx].fitlike_old = results[o][0]
                            #del results
                    print("duration =", time.time() - start)
                

                    #%% #Burn in start
                    print('multicore ray burn in phase')
                    burncount = 0
                    while burncount < len(weights):   
                    
                        for nw in range(len(weights)):
                            wo = weights[nw]; wv = weights[nw]; 

                            if Ds[nw].burn == True:
                                print(  'iteration start= ',time.strftime("%Y-%m-%d %H:%M:%S")); start = time.time()
                                
                                Ds[nw].std_cal(); # print(D.Xstd) # compute stds for each parameter for pCR adaption 
                                Ds[nw].propgen()  # obtaining a new set of proposed parameter values 
                                
                                if "EC" in comps:  
                                    for i in range(int(Ds[nw].nc/nprocs)):
                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            result_ids.append(ec_models[idx].update_params.remote(Ds[nw].chains[idx].proposal, comps,  ini_cons = ini_cons))
                                        results = ray.get(result_ids)
                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            result_ids.append(ec_models[idx].simulation.remote())
                                        results = ray.get(result_ids)
                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            result_ids.append(ec_models[idx].loglikelihood.remote(Ds[nw].chains[idx],dirNames[nw], start = False))
                                        results_ec = ray.get(result_ids)     
                                                
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            Ds[nw].chains[idx].Lnew    =  results_ec[o][0]*likered + results_ec[o][4] * wv + results_ec[o][5]* wo  
                                            Ds[nw].chains[idx].rmse_new = results_ec[o][2]; Ds[nw].chains[idx].sse_new = results_ec[o][3]
                                            Ds[nw].chains[idx].difv_new =  results_ec[o][4]
                                            Ds[nw].chains[idx].difo_new =  results_ec[o][5]; 
                                            Ds[nw].chains[idx].fitlike_new = results_ec[o][0]

                                    
                                if "T" in comps:  
                                    for i in range(int(Ds[nw].nc/nprocs)):
                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            result_ids.append(T_models[idx].update_params.remote(Ds[nw].chains[idx].proposal, comps,  ini_cons = ini_cons))
                                        results = ray.get(result_ids)
                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            result_ids.append(T_models[idx].simulation.remote())
                                        results = ray.get(result_ids)
                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            result_ids.append(T_models[idx].loglikelihood.remote(Ds[nw].chains[idx],dirNames[nw], start = False))
                                        results = ray.get(result_ids)     
                                                
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            if "EC" in comps and "T" in comps:
                                                Ds[nw].chains[idx].Lnew    +=  results[o][0] #+ results[o][4] * wv + results[o][5]* wo             
                                                Ds[nw].chains[idx].rmse_new += results[o][2]; 
                                                Ds[nw].chains[idx].sse_new += results[o][3]     
                                                Ds[nw].chains[idx].fitlike_new += results[o][0]

                                            else:
                                                Ds[nw].chains[idx].Lnew    =  results[o][0] + results[o][4] * wv + results[o][5]* wo  
                                                
                                                Ds[nw].chains[idx].rmse_new= results[o][2]; Ds[nw].chains[idx].sse_new = results[o][3]
                                                Ds[nw].chains[idx].difv_new =  results[o][4]
                                                Ds[nw].chains[idx].difo_new =  results[o][5]; 
                                                Ds[nw].chains[idx].fitlike_new = results[o][0]

                            
                                for i in range(Ds[nw].nc):     
                                    Ds[nw].chains[i].tprob() 
                                    Ds[nw].J[Ds[nw].chains[i].idd]    += np.nansum ( ( Ds[nw].chains[i].jump / Ds[nw].Xstd )**2  )
                                    Ds[nw].n_id[Ds[nw].chains[i].idd] += 1   
                                    Ds[nw].nacc   += Ds[nw].chains[i].accept 
                        
                                Ds[nw].update_R_hats(dirNames[nw], R_hat)
                                
                                if Ds[nw].ct != 0 and Ds[nw].ct >= 5: 
                                    Ds[nw].Chain_removal( Ds[nw].tc); # omega calculated using the last half of all likelihoods   
                                    Ds[nw].pCR_update();   # print(D.pCR)   # update selection probabilities of 
                                    # smaller idd vlaues mean smaller threshold CR values and thus a larger subset of samples (less pars to be kept)
                                    
                                if  Ds[nw].ct != 0 and Ds[nw].ct >= 5:  print('acceptance rate', Ds[nw].nacc /((Ds[nw].ct)* Ds[nw].nc) )  
                            
                                Ds[nw].print_recovery(dirNames[nw])
                                Ds[nw].update_current(dirNames[nw]) # updates current.dat file with current state of chains
                                Ds[nw].update_pars(dirNames[nw])

                                gc.collect()

                                if max(Ds[nw].R) < 1.2 and Ds[nw].ct > int(Ds[nw].rac/100*Ds[nw].nc): Ds[nw].burn = False 
                                #if max(Ds[nw].R) < 1.2: Ds[nw].burn = False 
                                if postrun: 
                                      if Ds[nw].ct == 5: Ds[nw].burn = False
                                print("duration =", time.time() - start)
                        
                        burncount = 0
                        for nw in range(len(weights)):
                            if Ds[nw].burn == False: 
                                burncount += 1
                                print(burncount, 'models converged')
                                print('----------------------------------')
                        
                        if  Ds[0].ct != 0 and Ds[0].ct%20 == 0 and len(weights) > 2:
                            #postpars, folder ,result_df= process_folders(directory = dirName_run, filename = "current_ext.dat", postest = False)
                            #smooth_l  = plt_Lcurve(subset =result_df,dirName = dirName_run, erfac = 10, showplt = False) # estimates a smooth L-curve 
                            #cur_w =estimate_w(result_df, smooth_l, dirName = dirName_run, mco = "MCO_02") # using curvature with respect to w 
                            find_weight(directory = dirName_run, erfac =10,  postest = False,  mco = "MCO_02", printres = False)

                
                    print('burn in for all weights completed. Sampling the posterior..')
                    print('###########################################################')
                    

                    for nw in range(len(weights)):
                        wo = weights[nw]; wv = weights[nw]; 

                        print("model", nw, "burn in finished after", Ds[nw].ct, "iterations")
                        Ds[nw].nb = Ds[nw].ct  # keeps track on how many iterations were needed to achieve convergence
                        # End of BURN-IN -----------------------------------------------------           
                        
                        #%% # Sampling of posterior starts ---------------------------------------------   
                        Ds[nw].ct = 0; Ds[nw].nacc = 0
                        
                        for i in range(Ds[nw].nc):
                            Ds[nw].chains[i].postpars     = np.append(np.append(np.append(np.append(np.append( np.append(np.copy(Ds[nw].chains[i].current), np.copy(Ds[nw].chains[i].rmse_new)), np.copy(Ds[nw].chains[i].sse_new)), np.copy(Ds[nw].chains[i].fitlike_new)), np.copy(Ds[nw].chains[i].difo_new)), np.copy(Ds[nw].chains[i].difv_new)),0)    # current parameters will be stored in pars
                            Ds[nw].chains[i].postlike     =  [Ds[nw].chains[i].Lold]
                        print('multicore ray posterior sampling phase')
                        
                        
                        while Ds[nw].ct < Ds[nw].rac:
                            Ds[nw].propgen()
                            
                            if "EC" in comps:  
                
                                for i in range(int(Ds[nw].nc/nprocs)):
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(ec_models[idx].update_params.remote(Ds[nw].chains[idx].proposal, comps,  ini_cons = ini_cons))
                                    results = ray.get(result_ids)
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(ec_models[idx].simulation.remote())
                                    results = ray.get(result_ids)
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(ec_models[idx].loglikelihood.remote(Ds[nw].chains[idx],dirNames[nw], start = False))
                                    results_ec = ray.get(result_ids)     
                                    for o in range (nprocs):
                                        idx = int(i*nprocs+o)
                                        Ds[nw].chains[idx].Lnew    =  results_ec[o][0]*likered + results_ec[o][4] * wv + results_ec[o][5]* wo             
                                        Ds[nw].chains[idx].rmse_new= results_ec[o][2]; Ds[nw].chains[idx].sse_new = results_ec[o][3]
                                        Ds[nw].chains[idx].difv_new =  results_ec[o][4]
                                        Ds[nw].chains[idx].difo_new =  results_ec[o][5]; 
                                        Ds[nw].chains[idx].fitlike_new = results_ec[o][0]

                                #del results_ec  

                            if "T" in comps:  
                                for i in range(int(Ds[nw].nc/nprocs)):
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(T_models[idx].update_params.remote(Ds[nw].chains[idx].proposal, comps,  ini_cons = ini_cons))
                                    results = ray.get(result_ids)
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(T_models[idx].simulation.remote())
                                    results = ray.get(result_ids)
                                    
                                    result_ids = []
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        result_ids.append(T_models[idx].loglikelihood.remote(Ds[nw].chains[idx],dirNames[nw], start = False))
                                    results = ray.get(result_ids)    
                                                            
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        if "EC" in comps and "T" in comps:
                                            Ds[nw].chains[idx].Lnew    +=  results[o][0] #+ results[o][4] * wv + results[o][5]* wo             
                                            Ds[nw].chains[idx].rmse_new += results[o][2]; 
                                            Ds[nw].chains[idx].sse_new += results[o][3]   
                                            Ds[nw].chains[idx].fitlike_new += results[o][0]

                                        else:
                                            Ds[nw].chains[idx].Lnew    =  results[o][0] + results[o][4] * wv + results[o][5]* wo             
                                            Ds[nw].chains[idx].rmse_new= results[o][2]; Ds[nw].chains[idx].sse_new = results[o][3]
                                            Ds[nw].chains[idx].difv_new =  results[o][4]
                                            Ds[nw].chains[idx].difo_new =  results[o][5];  
                                            Ds[nw].chains[idx].fitlike_new = results[o][0]

                                #del results
                                gc.collect()
                                                    
                            for i in range(int(Ds[nw].nc/nprocs)):
                                for o in  range (nprocs):
                                    idx = int(i*nprocs+o)
                                    Ds[nw].chains[idx].tprob()        
                                
                                if postsim:
                                    if "T" in comps:  

                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            if Ds[nw].chains[idx].accept == 1:
                                                result_ids.append(T_models[idx].v_o_int.remote())
                                            else:
                                                result_ids.append(T_models[idx].dummy_fun.remote())
                                        results_vv = ray.get(result_ids)


                                    if "EC" in comps:  
            
                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            if Ds[nw].chains[idx].accept == 1:
                                                result_ids.append(ec_models[idx].tau_cal.remote())
                                            else:
                                                result_ids.append(ec_models[idx].dummy_fun.remote())
                                        tau_results_ec = ray.get(result_ids)
            
                                        result_ids = []
                                        for o in  range (nprocs):
                                            idx = int(i*nprocs+o)
                                            if Ds[nw].chains[idx].accept == 1:
                                                result_ids.append(ec_models[idx].v_o_int.remote())
                                            else:
                                                result_ids.append(ec_models[idx].dummy_fun.remote())
                                        results_vv_ec = ray.get(result_ids)
                                        
            
            
                                    for o in  range (nprocs):
                                        idx = int(i*nprocs+o)
                                        if Ds[nw].chains[idx].accept == 1 and Ds[nw].nacc == 0:
                                            if "T" in comps:  
                                                Tcmoddf = pd.DataFrame(results[o][9])
                                                Tcmoddf['cmea'] = results[o][7]
                                                Tcmoddf['c_in'] =results[o][8]
                                                Tcmoddf['c_ou'] =results[o][10]
                                                
                                                Tvdf = pd.DataFrame(results_vv[o][0])
                                                Tddf = pd.DataFrame(results_vv[o][0])

                                            if "EC" in comps:  
                                                ecmoddf = pd.DataFrame(results_ec[o][9])
                                                ecmoddf['cmea'] = results_ec[o][7]
                                                ecmoddf['c_in'] = results_ec[o][8]
                                                evdf   = pd.DataFrame(results_vv_ec[o][0])
                                                odf    = pd.DataFrame(results_vv_ec[o][0])   
                                                etaudf = pd.DataFrame(tau_results_ec[o][1])
                                                
                                        if Ds[nw].chains[idx].accept == 1:                        
                                            Ds[nw].nacc   += Ds[nw].chains[idx].accept 
                                            if "T" in comps:  
                                                Tvdf['sim' + str(Ds[nw].nacc)] = results_vv[o][1]
                                                Tddf['sim' + str(Ds[nw].nacc)] = results_vv[o][2]
                                                Tcmoddf['sim' + str( Ds[nw].nacc)] = results[o][1]

                                            if "EC" in comps:  
                                                evdf['sim' + str(Ds[nw].nacc)] = results_vv_ec[o][1]
                                                odf['sim' + str(Ds[nw].nacc)] = results_vv_ec[o][2]
                                                etaudf['sim' + str(Ds[nw].nacc)] = tau_results_ec[o][0]
                                                ecmoddf['sim' + str( Ds[nw].nacc)] = results_ec[o][1]
                                                 
            


                            if Ds[nw].ct > 10 and Ds[nw].ct%5  == 0: print(Ds[nw].ct,Ds[nw].R)
                            if Ds[nw].ct != 0 and Ds[nw].ct%20 == 0: print('acceptance rate', Ds[nw].nacc /((Ds[nw].ct)* Ds[nw].nc ))
                        
                            Ds[nw].update_current( dirNames[nw]) 
                            Ds[nw].update_R_hats(dirNames[nw], R_hat)
                            Ds[nw].update_pars(dirNames[nw])

                            
                        Ds[nw].print_postpars(dirNames[nw])
                        
                        
                        if postsim:

                            foldname4  = dirNames[nw]
                            rst      = np.loadtxt(foldname4+'/postpars.dat' )
                            rstpd = pd.DataFrame(rst)
                            result_df = rstpd 
                            

                            bestres = np.array(result_df.nlargest(n=1, columns=[0]))
                            bestparam = bestres[0][1:npars+1]

                            if "T" in comps:  
                                Tcmoddf.to_csv(os.path.join(foldname4) +  '/'+ "T_cmod_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')
                                Tvdf.to_csv(os.path.join(foldname4) +  '/'+ "T_v_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.5f')
                                Tddf.to_csv(os.path.join(foldname4) +  '/'+ "T_d_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.6f')
                    
                                T_models[0].update_params.remote(bestparam, comps,   ini_cons = ini_cons)
                                T_models[0].simulation.remote()
                                like, cmod, rmse, sse, vvdlike, oodlike, parlike, cmea, lbc, modtime ,rbcT   = ray.get( T_models[0].loglikelihood.remote(Ds[nw].chains[0],dirNames[nw], start = False))
                                #tau_vec, timev,tau_vec2 = ray.get( T_models[0].tau_cal.remote())
                                modt, vvs, oos = ray.get( T_models[0].v_o_int.remote())
                                
                    
                                f = open(os.path.join(foldname4) +  '/'+ "T_cmod_ml" + '.txt','w')
                                dim = np.shape(cmod)
                                for k in range(dim[0]):
                                    f.write('%g %g %g \n' %  (modtime[k], cmea[k] , cmod[k]) )        
                                f.write('\n')
                                f.close()
                                
                                f = open(os.path.join(foldname4) +  '/'+ "T_v_ml" + '.txt','w')
                                dim = np.shape(modt)
                                for k in range(dim[0]):
                                    f.write('%g %g \n' %  (modt[k], vvs[k]) )        
                                f.write('\n')
                                f.close()

                                f = open(os.path.join(foldname4) +  '/'+ "T_d_ml" + '.txt','w')
                                dim = np.shape(modt)
                                for k in range(dim[0]):
                                    f.write('%g %s \n' %  (modt[k],   str('{:04.5f}'.format(oos[k]) )     ) )        

                                f.write('\n')
                                f.close()
                                

                            if "EC" in comps:  
                                
                                etaudf.to_csv(os.path.join(foldname4) +  '/'+ "ec_tau_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')
                                ecmoddf.to_csv(os.path.join(foldname4) +  '/'+ "ec_cmod_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')
                                evdf.to_csv(os.path.join(foldname4) +  '/'+ "ec_v_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')
                                odf.to_csv(os.path.join(foldname4) +  '/'+ "oo_all" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')
                                
                                ec_models[0].update_params.remote(bestparam, comps,   ini_cons = ini_cons)
                                ec_models[0].simulation.remote()
                                like, cmod, rmse, sse, vvdlike, oodlike, parlike, cmea, lbc, modtime, rbcEC    = ray.get( ec_models[0].loglikelihood.remote(Ds[nw].chains[0],dirNames[nw], start = False))
                                tau_vec, timev, tau_vec2 = ray.get( ec_models[0].tau_cal.remote())
                                modt, vvs, oos = ray.get( ec_models[0].v_o_int.remote())
                                
                    
                                f = open(os.path.join(foldname4) +  '/'+ "ec_cmod_ml" + '.txt','w')
                                dim = np.shape(cmod)
                                for k in range(dim[0]):
                                    f.write('%g %g %g \n' %  (modtime[k], cmea[k] , cmod[k]) )        
                                f.write('\n')
                                f.close()
                                
                                f = open(os.path.join(foldname4) +  '/'+ "ec_v_ml" + '.txt','w')
                                dim = np.shape(modt)
                                for k in range(dim[0]):
                                    f.write('%g %g \n' %  (modt[k], vvs[k]) )        
                                f.write('\n')
                                f.close()
                                
                                f = open(os.path.join(foldname4) +  '/'+ "o_ml" + '.txt','w')
                                #dim = np.shape(oo_ts)
                                dim = np.shape(modt)
                                for k in range(dim[0]):
                                    f.write('%g %g \n' %  (modt[k], oos[k]) )        
                                f.write('\n')
                                f.close()                                
                                
                                f = open(os.path.join(foldname4) +  '/'+ "ec_tau_ml" + '.txt','w')
                                dim = np.shape(tau_vec)
                                for k in range(dim[0]):
                                    f.write('%g %g \n' %  (timev[k], tau_vec[k]) )        
                                f.write('\n')
                                f.close()
                                
                        Ds[nw].print_Rhats(dirNames[nw], R_hat)
                        Ds[nw].print_DREAM_settings(dirNames[nw])       
                                      
                    #%% # Postprocessing: determining optimal L curve weight     
                    #from regular_vtraflux import *
                    #find_weight(directory = dirName_run, erfac =10,  postest = True,  mco = "MCO_02", printres = True)
                    
                        
        ray.shutdown()
        assert ray.is_initialized() == False    
                        
        
