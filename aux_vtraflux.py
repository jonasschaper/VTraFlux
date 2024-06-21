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
from vtraflux import *


# Dream settings (some are automatically loaded from recovery file upon Restart)------------------------------------
dream_delta    = 3; 
ncr            = 3;                 
maxsetback     = 250     
boundary       = "reflect" # choose from "bound", "folding" & "random" see Vrugt et al. 2016 (Environmental Modelling & Software)  Fig. 6
reset_removal  = False   # resets DREAM parameter upon restart if set to True

# ------------------------------------------------------



def DREAM_ecmod_create( site, comps, dat_info, par_info, run_info):
    
    workstation  = run_info["workstation"]
    supercomp    = run_info["supercomputer"]
    vantage      = run_info["vantage"]
    Restart      = run_info["Restart"]
    beta0        = run_info["beta0"]
    vantage_w    = run_info["vantage_w"]
    weights      = run_info["weights"]

    uninfp       = dat_info["priors"]
    chunk        = dat_info["chunk"]


    modfold      = run_info["modfold"]

    
    h_factor = par_info["v_int"]; off_typ = par_info["o_type"]; off_set = par_info["offset"]
    st_t     = par_info["st_t"]
    vtype    = par_info["v_type"]
    ndays = (chunk[1]-chunk[0])+1


    # set directories depending on which machine the code is running
    if workstation == True:
        mfolder      = modfold # "C:/Users/Jonas/Documents/tempts/transnum/"
        nchains = 28;
        if h_factor == 3: nchains = 48; # will crash the workstations
        if h_factor <= 3: nchains = 42;

    if supercomp == True:
        mfolder      =  modfold 
        nchains = 28;        
        if h_factor <= 3: nchains = 56;
        if par_info["v_type"] == "constant": nchains = 28
        if chunk[1] >= 12 and  h_factor <= 6: nchains = 56
    

    # dream results are stored in dreamres 
    resfold = mfolder  + "dreamres/"
    
    if not os.path.exists(resfold): os.mkdir(resfold); print("Directory " , resfold ,  " Created ")
    else:  print("Directory " , resfold ,  " already exists")
        

    prange = uninfp  
    

    if "EC" in comps:
        depthpair_EC = dat_info["depthpairs_EC"][0]

        locs_m_EC = depthpair2locs_m(depthpair_EC)
        
        if  off_typ == "constant" or off_typ == "none" :
            npars_offset = 1
        else:
            npars_offset = (len(locs_m_EC)-1)*ndays +1
        if off_set == False: npars_offset  -= 1
        
        print_depth = str( int( 100*(locs_m_EC[1]-locs_m_EC[0])))
        
    if "T" in comps:
        
        depthpair_T = dat_info["depthpairs_T"][0]
        locs_m_T = depthpair2locs_m(depthpair_T)
        if len(locs_m_T) == 3: f = 2
        if len(locs_m_T) == 2: f = 1
        print(depthpair_T)
        print_depth = str( int( 100*(locs_m_T[1]-locs_m_T[0]))) + str(int( 100*(locs_m_T[2])))

    off_set       = True
    if off_typ == "none": off_set = False 
    
    dfolder = mfolder  + "indata/" + site + "/"  # mfolder  + "indata/erpe19_01/" #nchunks = [(1,17)] # from day 1 to day 12
          
    startday = chunk[0]
    
    
    
    foldname =   Lcurve_fold_name_make(comps, par_info, chunk, print_depth, uninfp, site)
    dirName =    mfolder +  "dreamres/"  + foldname
    
    if not os.path.exists(dirName): os.mkdir(dirName); print("Directory " , dirName ,  " Created ")
    else:  print("Directory " , dirName ,  " already exists")
    
    dirName2s = []
    for w in weights:
        if w == 0: w = 1
        foldname_run =  fold_name_make(comps, par_info, w, chunk, print_depth, uninfp, site)
        dirName2 = dirName + "/" +  foldname_run
        
        if not os.path.exists(dirName2): os.mkdir(dirName2); print("Directory " , dirName2 ,  " Created ")
        else:  print("Directory " , dirName2 ,  " already exists")

        print_run_settings(dirName2, comps, par_info, w, dat_info, site)
        dirName2s.append(dirName2)
    

    # loading input data          
    if site == "syndata_01": 
        if "T" in comps:indata_T,   nt = synth_BC(locs_m_T, max_day = ndays, minutes = 5, solute = False,  prange = prange, vv_vel= 0.005, a_L=0.01)
        if "EC" in comps: indata_EC,  nt = synth_BC(locs_m_EC, max_day = ndays, minutes = 5, solute = True, prange = prange, vv_vel= 0.005, a_L=0.01)
    else:
        print(dfolder)
        if "EC" in comps: indata_EC = load_input_data(dfolder, depthpair_EC, 'EC' ,  locs_m_EC, ndays ); solute = True
        if "T" in comps:  indata_T  = load_input_daata(dfolder, depthpair_T, 'T' ,  locs_m_T, ndays ); solute = False
  
    rac  = int(25200/nchains) # should depend on numbers of parameters & acceptance rate

    logscale, uniform, Mean, width, rmin, rmax = load_priors(ndays, comps , param_info = par_info, prange = prange)
    npars = len(Mean); print("number of pars determined from prior-vec lengths:", npars)
    
    
    Ds = [DREAM(nchains, npars, maxsetback, rac, beta0 = beta0, boundary = boundary,ncr = ncr, delta = dream_delta,  ID = i) for i in range(len(weights))]

    for i in range(len(weights)):
        
    # creating a set of start parameters or recover parameters from unfinished runs
        Ds[i].par_set(logscale,uniform,Mean,width,rmin,rmax)
        Ds[i].restart(dirName, Restart) # recovers Lold, currents, and DREAM run stats (D.ct, D.tc etc.)
    
    result_df = []

    if vantage:
        for w in weights:
            if w == 0: w = 1
            foldname_run =  fold_name_make(comps, par_info, w, chunk, print_depth, uninfp, site)
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





def Lcurve_fold_name_make(comps, param_info, chunk, print_depth, uninfpstr, site):


    #vvstr = str(param_info["v_int"])
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
    
    foldname_a =  print_depth+"_" +'pri'+ uninfpstr[0] #+"_" +'w' + str(round(((np.log10(w)) ),1)) 
    
    if "EC" in comps:
        v_info += o_info
    if comps == "EC_T":
        compsstr = "EC.T"
    else:
        compsstr = comps
        
    foldname =  compsstr + "_" + site + '_' + str(chunk[0]) + '_' + str(chunk[1]) + '_' + v_info + "_" + foldname_a
    

    
    
    return foldname

def fold_name_make(comps, param_info, w, chunk, print_depth, uninfpstr, site):


    #vvstr = str(param_info["v_int"])
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
    
    foldname_a =  print_depth+"_" +'pri'+ uninfpstr[0]+"_" +'w' + str(round(((np.log10(w)) ),1)) 
    
    if "EC" in comps:
        v_info += o_info
    if comps == "EC_T":
        compsstr = "EC.T"
    else:
        compsstr = comps
        
    foldname =  compsstr + "_" + site + '_' + str(chunk[0]) + '_' + str(chunk[1]) + '_' + v_info + "_" + foldname_a
    

    
    
    return foldname

def load_tracer_data(datapath, depth_cm):
    f = open(datapath,'r') #opens the parameter file - r = read
    line = f.readlines()
    data = []
    for l in line:
        if l[0] != "#":
            data.append(l)
        #while l[0] != '#':
        #    l=f.readline()
    f.close() 
    
    line = 1
    NBOUND = int((data[0].strip().split())[0])
    
    
    USTIME = np.zeros(NBOUND)
    USBC = np.zeros(NBOUND)
    l = []


    for i in range(NBOUND) :
                USTIME[i] = float((data[line+i].strip().split())[0])     
                USBC[i] = float((data[line+i].strip().split())[1]) 
                l.append(  [depth_cm, USTIME[i],  USBC[i]] )
    return USTIME, USBC, l

def depthpair2locs_m(depthpair):
    
    locs_m = []
    for ii in range(0, (len( depthpair.split('_')))  ):   
        depth = float(depthpair.split('_')[ii])/100
        locs_m.append(depth)
    locs_m = np.array(locs_m) 
    
    return locs_m

def load_input_data(dfolder, depthpair, comp ,  locs_m , ndays):
    
    
    '''
    dfolder, depthpair, 'EC' , ndays, startday, locs_m, start_times
    locs_m_array = locs_m
    folder = dfolder
    
    comp = 'EC'
    day = 1
    '''

    if comp == 'EC': comp = "ec"

    input_dict = {}; dat = []
    cons_innl = []; time_innl = []; c_t_measl = []; c_t_measl2 = []

        
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
    
    #plt.plot(time_innl, cons_innl)
    len(time_innl)


    input_dict.update( {ndays : dat } )


    #return cmeasl, mask_obsl, tcmeasl, time_inn, cons_inn, nt, weightsl, cmeaslorg
    return input_dict

def print_run_settings(dirName, comps, par_info, w, dat_info, site):
                              
    
    uninfpstr       = dat_info["priors"]
    chunk        = dat_info["chunk"]

    
    h_factor = par_info["v_int"]; off_typ = par_info["o_type"]; off_set = par_info["offset"]
    st_t     = par_info["st_t"]
    vtype    = par_info["v_type"]
    
    depthpair_T = dat_info["depthpairs_T"][0]

    depthpair_EC = dat_info["depthpairs_EC"][0]

    
        #vvstr = str(param_info["v_int"])
    a= par_info["v_int"]
    vvstr =  f"{a:02}"
    if str(par_info["v_type"]) == "constant": vvstr = "00"
    
    #oostr = str(param_info["o_int"])
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
    
    #foldname_a =  print_depth+"_" +'pri'+ uninfpstr[0]+"_" +'w' + str(round(((np.log10(w)) ),1)) 
    
    if "EC" in comps:
        v_info += o_info
    if comps == "EC_T":
        compsstr = "EC.T"
    else:
        compsstr = comps
        
    #foldname =  compsstr + "_" + site + '_' + str(chunk[0]) + '_' + str(chunk[1]) + '_' + v_info + "_" + foldname_a
    
    
    f = open(dirName + '/' + 'run_info.dat','w')
    f.write('compsstr: '); f.write('%s ' % compsstr); f.write('\n')
    f.write('site: ');     f.write('%s ' % site ); f.write('\n')
    f.write('vvstr: ');    f.write('%s ' % vvstr); f.write('\n')
    f.write('oostr: ');    f.write('%s ' % oostr); f.write('\n')
    f.write('ststr: ');    f.write('%s ' % ststr); f.write('\n')
    f.write('priors: ');    f.write('%s ' % uninfpstr); f.write('\n')
    f.write('temp_depths: ');    f.write('%s ' % depthpair_T); f.write('\n')
    f.write('ec_depths: ');    f.write('%s ' % depthpair_EC); f.write('\n')
    f.write('ndays: ');    f.write('%g ' % (chunk[1])); f.write('\n')
    f.write('weight: ');    f.write('%g ' % (round(((np.log10(w)) ),1))  ); f.write('\n')
    f.close()