# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:06:54 2022

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

import math 
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy import interpolate
import random as rand

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# load vtraflux modules 
from dream_vtraflux import *
from priors_vtraflux   import *
from regular_vtraflux   import *
#from vtraflux import *



# ------------------------------------------------------



def DREAM_ecmod_create( site, comps, dat_info, par_info, run_info):

    
    global ncr
    global maxsetback
    global boundary
    global dream_delta

    # Dream settings (some are automatically loaded from recovery file upon Restart)------------------------------------
    dream_delta    = 3; 
    ncr            = 3;                 
    maxsetback     = 250     
    boundary       = "reflect" # choose from "bound", "folding" & "random" see Vrugt et al. 2016 (Environmental Modelling & Software)  Fig. 6

    
    #workstation  = run_info["workstation"]
    #supercomp    = run_info["supercomputer"]
    vantage      = run_info["vantage"]
    Restart      = run_info["recover"]
    sp           = run_info["savepars"]
    beta0        = run_info["beta0"]
    weights      = run_info["weights"]
    strt_opt     = run_info["strt_opt"]
    uninfp       = dat_info["priors"]
    chunk        = dat_info["chunk"]
    nchains      = run_info["nchains"]
    modfold      = run_info["modfold"]

    h_factor = par_info["v_int"]; off_typ = par_info["o_type"]; off_set = par_info["offset"]
    st_t     = par_info["st_t"]
    vtype    = par_info["v_type"]
    advective    = par_info["advective"]

    ndays = (chunk[1]-chunk[0])+1


    # set directories depending on which machine the code is running
    mfolder      = modfold 
    
    # dream results are stored in modres 
    resfold = mfolder  + "modres/"
    
    if not os.path.exists(resfold): os.mkdir(resfold); print("Directory " , resfold ,  " Created ")
    else:  print("Directory " , resfold ,  " already exists")
        

    prange = uninfp  
    

    if "EC" in comps:
        depthpair_EC = dat_info["depthpairs_EC"][0]

        locs_m_EC, print_depth = depthpair2locs_m(depthpair_EC)
        print_depth_EC = print_depth
        if  off_typ == "constant" or off_typ == "none" :
            npars_offset = 1
        else:
            npars_offset = (len(locs_m_EC)-1)*ndays +1
        if off_set == False: npars_offset  -= 1
        
        
    if "T" in comps:
        
        depthpair_T = dat_info["depths_T"][0]
        locs_m_T, print_depth = depthpair2locs_m(depthpair_T)
        if len(locs_m_T) == 3: f = 2
        if len(locs_m_T) == 2: f = 1
        print_depth_T= print_depth
    if "T" in comps and "EC" in comps:
        print_depth = print_depth_T + print_depth_EC

    off_set       = True
    if off_typ == "none": off_set = False 
    
    dfolder = mfolder  + "indata/" + site + "/"  
              
    
    
    foldname =   Lcurve_fold_name_make(comps, par_info, chunk, print_depth, uninfp, site, advective)
    dirName =    mfolder +  "modres/"  + foldname # folder where runs with different weighting factors are stored 
    
    if not os.path.exists(dirName): os.mkdir(dirName); print("Directory " , dirName ,  " Created ")
    else:  print("Directory " , dirName ,  " already exists")
    
    dirName2s = []
    for w in weights:
        if w == 0: w = 1
        foldname_run =  fold_name_make(comps, par_info, w, chunk, print_depth, uninfp, site, advective)
        dirName2 = dirName + "/" +  foldname_run
        
        if not os.path.exists(dirName2): os.mkdir(dirName2); print("Directory " , dirName2 ,  " Created ")
        else:  print("Directory " , dirName2 ,  " already exists")

        print_run_settings(dirName2, comps, par_info, w, dat_info, site)
        dirName2s.append(dirName2)
    

    # loading input data          
    if site == "syndata_01": 
        if "T" in comps:indata_T,   nt = synth_BC(locs_m_T, max_day = ndays, minutes = 5, solute = False,  prange = prange, vv_vel= 0.05, a_L=0.01)
        if "EC" in comps: indata_EC,  nt = synth_BC(locs_m_EC, max_day = ndays, minutes = 5, solute = True, prange = prange, vv_vel= 0.05, a_L=0.01)
    else:
        print(dfolder)
        if "EC" in comps: indata_EC = load_input_data(dfolder, depthpair_EC, 'EC' ,  locs_m_EC, ndays ); #solute = True
        if "T" in comps:  indata_T  = load_input_data(dfolder, depthpair_T, 'T' ,  locs_m_T, ndays ); #solute = False
    rac  = int(25200/nchains) # should depend on numbers of parameters & acceptance rate

    uniform, Mean, width, rmin, rmax, strt = load_priors(ndays, comps , param_info = par_info, prange = prange)
    npars = len(Mean); print("number of pars determined from prior-vec lengths:", npars)
    
    
    Ds = [DREAM(nchains, npars, maxsetback, rac, beta0 = beta0, boundary = boundary,ncr = ncr, delta = dream_delta, sp=sp, ID = i) for i in range(len(weights))]

    for i in range(len(weights)): # creating a set of start parameters or recover parameters from unfinished runs
        Ds[i].par_set(uniform,Mean,width,rmin,rmax,strt, strt_opt = strt_opt)
        Ds[i].restart(dirName2s[i], Restart) # recovers Lold, currents, and DREAM run stats (D.ct, D.tc etc.)
    
    result_df = []
    if vantage:
        for w in weights:
            if w == 0: w = 1
            foldname_run =  fold_name_make(comps, par_info, w, chunk, print_depth, uninfp, site, advective)
            
            if run_info["restart_vantage"]:
                wlist = run_info["vantage_wl"]
                w = wlist[weights.index(w)]
                foldname_run =  fold_name_make(comps, par_info, w, chunk, print_depth, uninfp, site, advective)

            dirName2 = dirName + "/" +  foldname_run
                     
            print(dirName2+'/current.dat' )
            rst      = np.loadtxt(dirName2 +'/current.dat' ) 
            rstpd = pd.DataFrame(rst)
            result_df.append(rstpd )
    else :
            result_df = 99
    
    if "EC" in comps and "T" in comps:
        indata = [indata_EC[ndays], indata_T[ndays]]
    elif "EC" in comps:
        indata = indata_EC[ndays]
    elif "T" in comps:
        indata = indata_T[ndays]

    if "EC" in comps and "T" in comps:
        locs_m = [locs_m_EC, locs_m_T]
    elif "EC" in comps:
        locs_m = locs_m_EC
    elif "T" in comps:
        locs_m = locs_m_T
    
    return Ds,  dirName2s, indata, result_df, npars, locs_m, dirName



def Lcurve_fold_name_make(comps, param_info, chunk, print_depth, uninfpstr, site, advective):

    a= param_info["v_int"]
    vvstr =  f"{a:02}"
    if str(param_info["v_type"]) == "constant": vvstr = "00"
    
    #oostr = str(param_info["o_int"])
    if param_info["offset"] == True:
        b= param_info["o_int"]
        oostr = param_info["o_type"][0]+ f"{b:02}"
        
        if str(param_info["o_type"]) == "constant": oostr = param_info["o_type"][0]+ "00"
        if str(param_info["o_type"]) == "linear": oostr = "lt2"
        
    else:
        oostr = "nn"
    
    c = param_info["st_t"]
    ststr =  f"{c:02}"
    
    v_info = "v" + param_info["v_type"][0]+ vvstr + "st" + ststr
    o_info =  "_o" + oostr
    reldiff = str(param_info["reldiff"])[0]

    if param_info["reldiff"]:
        reldiff = "T"

    foldname_a =  print_depth+  "_" + reldiff+ advective[0] +"_"+'pri'+ uninfpstr[0] #+"_" +'w' + str(round(((np.log10(w)) ),1)) 
    
    if "EC" in comps:
        v_info += o_info
    if comps == "EC_T":
        compsstr = "EC.T"
    else:
        compsstr = comps
    foldname =  compsstr + "_" + site + '_' + str(chunk[0]) + '_' + str(chunk[1]) + '_' + v_info + "_" + foldname_a
    
    return foldname

def fold_name_make(comps, param_info, w, chunk, print_depth, uninfpstr, site, advective):


    a= param_info["v_int"]
    vvstr =  f"{a:02}"
    if str(param_info["v_type"]) == "constant": vvstr = "00"
    
    if param_info["offset"] == True:
        b= param_info["o_int"]
        oostr = param_info["o_type"][0]+ f"{b:02}"
        
        if str(param_info["o_type"]) == "constant": oostr = param_info["o_type"][0]+ "00"

        if str(param_info["o_type"]) == "linear": oostr = "lt2"
        
    else:
        oostr = "nn"
    
    c = param_info["st_t"]
    ststr =  f"{c:02}"

    reldiff = str(param_info["reldiff"])[0]
    v_info = "v" + param_info["v_type"][0]+ vvstr + "st" + ststr
    o_info =  "_o" + oostr
    
    if param_info["reldiff"]:
        reldiff = "T"
    
    foldname_a =  print_depth+"_" +reldiff+ advective[0] +"_" +'pri'+ uninfpstr[0]+"_" +'w' + str(round(((np.log10(w)) ),1)) 
    
    if "EC" in comps:
        v_info += o_info
    if comps == "EC_T":
        compsstr = "EC.T"
    else:
        compsstr = comps
        
    foldname =  compsstr + "_" + site + '_' + str(chunk[0]) + '_' + str(chunk[1]) + '_' + v_info + "_" + foldname_a
    
    return foldname

def load_tracer_data(datapath, depth_cm):
    f = open(datapath,'r') 
    line = f.readlines()
    data = []
    for l in line:
        if l[0] != "#":
            data.append(l)
    f.close() 
        
    line = 1
    nbound = int((data[0].strip().split())[0])
    uptime = np.zeros(nbound)
    upbc = np.zeros(nbound)
    l = []

    for i in range(nbound) :
                uptime[i] = float((data[line+i].strip().split())[0])     
                upbc[i] = float((data[line+i].strip().split())[1]) 
                l.append(  [depth_cm, uptime[i],  upbc[i]] )
    return uptime, upbc, l

def depthpair2locs_m(depthpair):
    
    locs_m = []; printdepth = ""
    for ii in range(0, (len( depthpair.split('_')))  ):   
        depth = float(depthpair.split('_')[ii])/100
        locs_m.append(depth)
        printdepth = printdepth + depthpair.split('_')[ii]
    locs_m = np.array(locs_m) 
    
    return locs_m, printdepth

def load_input_data(dfolder, depthpair, comp ,  locs_m , ndays):
    
    if comp == 'EC': comp = "ec"

    input_dict = {}; dat = []

        
    sw_fold = dfolder + comp +  "_" +  depthpair.split('_')[0] + ".txt"
    time_inn, cons_inn, conc_time  = load_tracer_data(sw_fold, locs_m[0])
    
        
    pw_fold = dfolder + comp +   "_" +  depthpair.split('_')[1] + ".txt"
    t_cmeas,     cmeas, pw_conc_time     = load_tracer_data(pw_fold, locs_m[1])

    if len(locs_m) == 3:
        pw_fold = dfolder + comp +   "_" +  depthpair.split('_')[-1] + ".txt"
        t_cmeas,     cmeas2, pw_conc_time     = load_tracer_data(pw_fold, locs_m[-1])
            
    dat.append([locs_m[0], np.array(time_inn),  np.array(cons_inn)])
    dat.append([locs_m[1], np.array(time_inn),  np.array(cmeas)])
    if len(locs_m) == 3:
        dat.append([locs_m[-1], np.array(time_inn),  np.array(cmeas2)])
    

    input_dict.update( {ndays : dat } )
    return input_dict

def print_run_settings(dirName, comps, par_info, w, dat_info, site):
                              
    uninfpstr  = dat_info["priors"]
    chunk      = dat_info["chunk"]

    depthpair_T = dat_info["depths_T"][0]
    depthpair_EC = dat_info["depthpairs_EC"][0]

    a= par_info["v_int"]
    vvstr =  f"{a:02}"
    if str(par_info["v_type"]) == "constant": vvstr = "00"
    
    if par_info["offset"] == True:
        b= par_info["o_int"]
        oostr = par_info["o_type"][0]+ f"{b:02}"
        
        if str(par_info["o_type"]) == "constant": oostr = par_info["o_type"][0]+ "00"

        if str(par_info["o_type"]) == "linear": oostr = "lt2"
        
    else:
        oostr = "nn"
    
    c = par_info["st_t"]
    ststr =  f"{c:02}"
    v_info = "v" + par_info["v_type"][0]+ vvstr + "st" + ststr
    o_info =  "_o" + oostr
    

    if "EC" in comps:
        v_info += o_info
    if comps == "EC_T":
        compsstr = "EC.T"
    else:
        compsstr = comps
        
    f = open(dirName + '/' + 'run_info.dat','w')
    f.write('compsstr: '); f.write('%s ' % compsstr); f.write('\n')
    f.write('site: ');     f.write('%s ' % site ); f.write('\n')
    f.write('vvstr: ');    f.write('%s ' % vvstr); f.write('\n')
    f.write('oostr: ');    f.write('%s ' % oostr); f.write('\n')
    f.write('ststr: ');    f.write('%s ' % ststr); f.write('\n')
    f.write('priors: ');    f.write('%s ' % uninfpstr); f.write('\n')
    f.write('temp_depths: ');    f.write('%s ' % depthpair_T); f.write('\n')
    f.write('ec_depths: ');    f.write('%s ' % depthpair_EC); f.write('\n')
    f.write('ndays: ');    f.write('%g ' % (chunk[1]-chunk[0]+1)); f.write('\n')
    f.write('weight: ');    f.write('%g ' % (round(((np.log10(w)) ),1))  ); f.write('\n')
    f.write('TPO: ');    f.write('%s ' %     par_info["tpo"]); f.write('\n')
    f.write('T_err: ');    f.write('%s ' %     par_info["T_err"]); f.write('\n')
    f.write('EC_err: ');    f.write('%s ' %     par_info["EC_err"]); f.write('\n')
    f.write('reldiff: ');    f.write('%s ' %     par_info["reldiff"]); f.write('\n')
    f.close()
                       
def Stallman_1965(x, max_day, minutes, background_value, A , v, D ):
    
    t = np.linspace(0.,24*max_day, int(24*60*max_day/minutes))
    nt = int(24*60*max_day/minutes)
    P = 24
    tx_t = []
    
    for n in range(nt):
        tt = t[n]        
        alpha = np.sqrt(v**4 + ( (8*math.pi * D)/P  )**2   )
        exp_arg = (v*x)/(2*D) -  ((x)/(2*D)) * np.sqrt( (alpha+v**2)/2)
        cos_arg = (2*math.pi*tt)/(P) -  ((x)/(2*D)) * np.sqrt( (alpha-v**2)/2)
        value_xt = background_value+  A * np.exp(exp_arg)   * math.cos(cos_arg) 
        tx_t.append(round(value_xt,4))

    return np.array(tx_t)

def synth_BC(seclocs, max_day, minutes, solute, prange,  a_L, vv_vel ):
    nn   = 0.35   
    vvs  = vv_vel

    t    = np.linspace(0.,24*max_day, int(24*60*max_day/minutes))
    ka   = 2.0*3600 
    pcs   = 2.2*1e6 
    beta =  0.01
    overall_temp = []
    pcw = 4.184*1e6

    if solute == True:
        vvs = vv_vel
        dds  =  0.3e-9*3600*0.7 + vvs *  a_L
        t00_t = [0.0, t, Stallman_1965(x = 0.00, max_day = max_day, minutes = minutes, background_value = 1.000, A = 0.5 , v =  vvs, D = dds)]
        overall_temp.append(t00_t)

        for j in range(1,len(seclocs)):
            t08_t = [seclocs[j],t, Stallman_1965(x = seclocs[j], max_day = max_day, minutes = minutes, background_value = 1.000, A = 0.5 , v =  vvs, D = dds)]
            overall_temp.append(t08_t)

    elif solute == False:
        vvt   =  (vvs) * (pcw/(nn*pcw + (1-nn)*pcs)) *nn
        ddt   =  ka / (nn*pcw + (1-nn)*pcs)  +   vvt*beta 
        t00_t = [0.0, t, Stallman_1965(x = 0.00, max_day = max_day, minutes = 5, background_value = 15, A = 5 , v =  vvt, D = ddt)]
        overall_temp.append(t00_t)
        for j in range(1,len(seclocs)):
            t08_t = [seclocs[j],t, Stallman_1965(x = seclocs[j], max_day = max_day, minutes = 5, background_value =15, A = 5 , v =  vvt, D = ddt)]
            overall_temp.append(t08_t)
    data_dict= {}
    data_dict.update({ max_day:overall_temp})
    
    return data_dict, len(t)                    
                    