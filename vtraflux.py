# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 21:20:12 2022

@author: jonas schaper
"""
import ray
import numpy as np
import copy
from scipy.sparse import diags
from scipy import interpolate
#from lmfit import minimize, Parameters, report_fit
import time
import matplotlib.pyplot as plt
import os


@ray.remote(memory=250 * 1024 * 1024)
class vtraflux_model:
    """
    reactive_model class is a one dimsenisonal advection dispersion model
    that incorporates heat transport 
    """
    
    def __init__(self, indata,
                       chunk,
                       dt,
                       rbc,
                       seclocs,
                       param_info,
                       comp,
                       ID = None):
               
        self.comp       = comp
        self.st_t        = param_info["st_t"]  # model time at which first staging post is put
        self.restime_h   = param_info["st_t"]  # time from which measurments are considered in objective function        
        self.chunk = chunk
        self.sel_tvec = np.where(indata[0][1]  >24*(chunk[0]-1)) and np.where(indata[0][1]  <=24*chunk[1]) 
        
        # cuts out required days (could change to hours )
        
        self.wv = 0

        self.joint = False 
        self.therm_rfac = 0.4


        if comp == "T": 
            self.solute = False
            self.off_typ = "none"
            self.offset = False
        else:
            self.solute = True
            self.offset = param_info["offset"]
        
        if comp == "EC": 
            self.off_typ = param_info["o_type"]
                
        if comp == "EC_T":
            self.joint = True 
            self.solute = False
            self.off_typ = "none"
            self.offset = False
        
        self.v_type =  param_info["v_type"]
        
        if len(indata) == 3:
            rbc = indata[-1][-1][self.sel_tvec]
            self.ploc  = indata[1][0] -indata[0][0]
            self.x_max = indata[-1][0]-indata[0][0]
        else:
            self.ploc = indata[-1][0]-indata[0][0]



        self.dt     = dt        
        self.nt = int((indata[0][1][self.sel_tvec][-1]+ (indata[0][1][self.sel_tvec][1]-indata[0][1][self.sel_tvec][0]))/self.dt)
        self.indata = copy.deepcopy(indata)
        self.lbc         = indata[0][2][self.sel_tvec]
        self.cmea  = indata[1][2][self.sel_tvec]
        self.modtime    =  np.array(indata[1][1][self.sel_tvec])
        self.ndays = self.chunk[1]

        self.iter_count = 0
        self.stagingpost_position = "corner" # "mid 

        if isinstance(rbc, float) == 1:
            self.rbc_type    = "Neumann"
            self.rbc_intvals = np.ones( len(self.lbc))*(np.mean(self.lbc)+np.mean(self.cmea))/2 # constant groundwater value 
            
            #print("rbc = Neumann")
            self.x_max = (seclocs[-1]-seclocs[0])*2 + 0.3
        
        elif len(rbc) > 1:
            self.rbc_type    = "Dirichlet"
            self.rbc_intvals = copy.deepcopy(rbc[self.sel_tvec])
            #print("rbc = Dirichlet")
            if self.solute:
                self.x_max = (seclocs[-1]-seclocs[0])*4 + 0.3        


        self.dx_prop = 1e-3 # 0.5e-3 
        self.x_dis = np.linspace(0,self.x_max,int(self.x_max/self.dx_prop)+1)
        self.dx = self.x_dis[1]-self.x_dis[0]
               
        self.reldiff = True         # 210324 changed to relative parameter differences         

        
        self.rbc    = rbc

        self.ID            = ID
        self.Dmol          = 0.3e-9*3600 # default vlaue in PHREEQC
        self.cout          = np.zeros((len(indata[0][1]) ,len(seclocs)-1))
        self.advective     = "centered"   # "upwind_1" performace not as good as centered 
        self.time_dis      = "forward"  

        self.seclocs       = seclocs        

        self.vvs = 0.1
        self.dds =  0.001
        self.kks = 0.0
        self.rrs = 1
        self.pcw = 4.184*1e6   # J / (K  m3)
        self.y   = 0  # zero-order production rate 
        self.pcs = 3.0*1e6
        self.law = 0.59*3600 # W/3600 m-1 °C-1 thermal conductivity of water at 25 degC

        # staging posts for interpolation values: velocity  
        self.v_per_d = param_info["v_int"] 
        
        if self.v_type == "variable":
            #print('using a time-dependent porewater velocity')
            vvs_tt = [0,self.st_t]
            vstart = 1
            if self.st_t == 0:           
                vvs_tt = [self.st_t]
                vstart = 1

            for i in range(vstart,int(self.ndays*24/self.v_per_d-(self.st_t/self.v_per_d-1))  ):
                vvs_tt.append(self.st_t+self.v_per_d*i) # staging posts edges of interval  
            self.vvs_t = vvs_tt
            self.iter = int(len(self.vvs_t))
        
        if self.v_type == "constant":
            print('using one constant porewater velocity')
            self.vvs_t = [0,self.ndays*24]
            self.iter = int(len(self.vvs_t)-1)

        if self.offset:
            if  self.off_typ == "variable":
                self.h_per_o = param_info["o_int"]; 
                oos_tt = []
                for i in range(int(self.ndays*24/self.h_per_o )+1):
                    oos_tt.append(self.h_per_o*i) # staging posts edges of interval  
                self.oos_t = oos_tt  
            elif  self.off_typ == "constant" or self.off_typ == "linear":
                self.oos_t = [0,self.ndays*24]  
             
            self.oo_int = interpolate.PchipInterpolator(self.oos_t, 0.0*np.ones(len(self.oos_t)))# need to find closes cin value to that point in time t

        
        self.t_max = copy.deepcopy(self.indata[0][1][-1])
        
        self.vv_int = interpolate.PchipInterpolator(self.vvs_t, 0.1*np.ones(len(self.vvs_t)))# need to find closes cin value to that point in time t
        self.lbc_int = interpolate.PchipInterpolator(self.modtime, self.lbc)
        

        self.rbc_int = interpolate.PchipInterpolator(self.modtime, self.rbc_intvals)
        
        
        # make sure an inital condition is constructed 
        self.ini_cons = False
        self.ini_cond_make()

    def ini_cond_make(self):
        #  initial velocity and flow direction  

        v_current = (self.vv_int(0))
        
        if v_current > 0: 
            # flow direction is downward
            ploc =  self.ploc

            if   self.ini_cons:         
                # running model using the first velocity only to get initial condition 
                ini_t = self.st_t
                self.nt = int(ini_t/self.dt)
    
                # initial condition based on test run 
                self.vv_int = interpolate.PchipInterpolator(self.vvs_t[0:2],  (self.vvs[0:2])) # need to find closes cin value to that point in time t
                self.inc_int      = interpolate.PchipInterpolator(np.array([0,self.ploc,self.x_max]), np.array([self.lbc[0],self.cmea[0], self.rbc_intvals[0]]))# need to find closes cin value to that point in time t
                self.simulation()
                self.inc_int = interpolate.PchipInterpolator(self.x_dis, 1*(np.mean(self.Vout, axis = 0))   )# need to find closes cin value to that point in time t
                #self.inc_int = interpolate.PchipInterpolator(self.x_dis, 0.5*(self.inc_int_base(self.x_dis)+np.mean(self.Vout, axis = 0))   )# need to find closes cin value to that point in time t
                self.nt = int((self.indata[0][1][-1]+ (self.indata[0][1][1]-self.indata[0][1][0]))/self.dt)
                 
       
            else:
                # simple initial condition via interpolation
                
                if self.off_typ == "none":
                    self.inc_int      = interpolate.PchipInterpolator(np.array([0,self.ploc,self.x_max]), np.array([self.lbc[0],self.cmea[0], self.rbc_intvals[0]]))# need to find closes cin value to that point in time t
                    self.inc_int_base = interpolate.PchipInterpolator(np.array([0,self.ploc,self.x_max]), np.array([self.lbc[0],self.cmea[0], self.rbc_intvals[0]]))# need to find closes cin value to that point in time t
                    #print("lsssssalxxxxal")
                    
                if self.off_typ == "variable"or self.off_typ == "constant" or self.off_typ == "linear":

                    self.inc_int      = interpolate.PchipInterpolator(np.array([0,self.ploc, self.x_max]), np.array([self.lbc[0] , self.lbc[0]   , self.rbc_intvals[0]  ]))# need to find closes cin value to that point in time t
                    self.inc_int_base = interpolate.PchipInterpolator(np.array([0,self.ploc, self.x_max]), np.array([self.lbc[0] , self.lbc[0]   , self.rbc_intvals[0]  ]))# need to find closes cin value to that point in time t
                   
                    
        else: 
            # flow direction is downward, the domain is turned upside down 
            ploc = self.x_max - self.ploc
            if   self.ini_cons: 
                ini_t = self.st_t
                self.nt = int(ini_t/self.dt)
    
                # initial condition based on test run 
                self.vv_int = interpolate.PchipInterpolator(self.vvs_t[0:2],  (self.vvs[0:2])) # need to find closes cin value to that point in time t
                
                self.inc_int      = interpolate.PchipInterpolator(np.array([0,ploc,self.x_max]), np.array([self.rbc_intvals[0],self.cmea[0], self.lbc[0]]))# need to find closes cin value to that point in time t

                self.simulation()
                self.inc_int = interpolate.PchipInterpolator(self.x_dis, 1*(np.mean(self.Vout, axis = 0))   )# need to find closes cin value to that point in time t
                self.nt = int((self.indata[0][1][-1]+ (self.indata[0][1][1]-self.indata[0][1][0]))/self.dt)
                #print("lalsssssal")
            
            else:
                # simple initial condition via interpolation
                if self.off_typ == "none":
    
                    self.inc_int      = interpolate.PchipInterpolator(np.array([0,ploc,self.x_max]), np.array([self.rbc_intvals[0],self.cmea[0], self.lbc[0]]))# need to find closes cin value to that point in time t
                    self.inc_int_base = interpolate.PchipInterpolator(np.array([0,ploc,self.x_max]), np.array([self.rbc_intvals[0],self.cmea[0], self.lbc[0]]))# need to find closes cin value to that point in time t

                if self.off_typ == "variable" or self.off_typ == "constant" or self.off_typ == "linear":
                    self.inc_int = interpolate.PchipInterpolator(np.array([0, ploc, self.x_max]), np.array([self.rbc_intvals[0] , self.rbc_intvals[0]  , self.lbc[0]  ]))# need to find closes cin value to that point in time t
        

    def flow_direction(self):
        #  initial velocity and flow direction  

        if (self.vv_int(0)) > 0: 
            flow_dir_pre = "down"
            ploc =  self.ploc
        else: 
            flow_dir_pre = "up"
            # the domain is flipped or turned upside down, respectively 
            ploc = self.x_max - self.ploc
        return flow_dir_pre, ploc
        

    def simulation(self ):
        
        cinl = [];Vout  = []; modt = []; solsol = [] ; Dhl =[]       # list for storing V arrays at certain time steps

        self.dx = self.find_dx_Pe()
        self.x_dis = np.linspace(0,self.x_max,int(self.x_max/self.dx)+1)

        flow_dir_pre, ploc = self.flow_direction() #  initial velocity and flow direction  
        self.c_in = self.inc_int(self.x_dis)
        V        = copy.deepcopy(self.c_in)                  
        
        for n in range(0,self.nt-1):  

            modt.append(n+1)            
            
            v_current = (self.vv_int(n*self.dt))
            
            if self.solute:
                Dh = 0.7 * self.Dmol + abs(v_current)*self.a_L
            else:
                #if self.joint:
                #v_current_real = v_current/((self.pcw/(self.n*self.pcw + (1-self.n)*self.pcs) )*self.n)
                # thermal front velocity according to rau 2014 heat review
                # v_current is alwasy thermal front velocity 
                Dh = self.dds + abs(v_current)*(self.a_L**1)
            Dhl.append(Dh)
            
            self.create_matrices(abs(v_current), Dh)
            
            if v_current > 0:
                flow_dir = "down"
                ploc = self.ploc
                lbc_n  = float(self.lbc_int(n*self.dt))
                rbc_n  = float(self.rbc_int(n*self.dt))
                lbc_n1 = float(self.lbc_int( (n+1)*self.dt   ))# determine current inflow concentration
                rbc_n1  = float(self.rbc_int( (n+1)*self.dt    ))
                if flow_dir_pre is not flow_dir: 
                    V        = copy.deepcopy(V[::-1]) 
            else:
                flow_dir = "up"
                if flow_dir_pre is not flow_dir: 
                    V        = copy.deepcopy(V[::-1]) 

                ploc = self.x_max - self.ploc
                rbc_n  = float(self.lbc_int(n*self.dt))
                lbc_n  = float(self.rbc_int(n*self.dt))
                rbc_n1 = float(self.lbc_int( (n+1)*self.dt   ))
                lbc_n1  = float(self.rbc_int( (n+1)*self.dt    ))                
            
            Vn      = V
            B       = np.dot(self.Bsolo,Vn[1:-1]) # from position c 1 to c m-1, -1 excludes last element because ":" refers to range
            
            if self.time_dis == "forward" and self.advective == "centered":
                B[0]=B[0]+ (  (self.lambdas+self.alphas)*lbc_n1) + (lbc_n*(self.lambdas+self.alphas)) 
                if self.rbc_type == "Dirichlet": # updating upstream Dirichlet BC (c_0[n] and c_0[n+1])
                    B[-1]=B[-1]  +((self.ls_lam-self.ls_alp)*rbc_n1  +rbc_n*(self.ls_lam-self.ls_alp))       
                elif self.rbc_type == "Neumann": # updating Dirichlet or Neumann BC at c_m[n] and c_m[n+1] 
                    B[-1]=B[-1] +V[-1]*(self.ls_lam-self.ls_alp) +(1/3)*(rbc_n*(self.ls_lam-self.ls_alp)*2*self.dx)  
                
            B = B + self.y*self.dt # adding a zero order production rate 

            V[1:-1] = np.linalg.solve(self.A,B)  
            
            # from here on, V stores the solution of the n+1 timestep 
            # both the V[0] aad V[-1 = m] need to be updated
            cinl.append(lbc_n1)
            V[0]    = lbc_n1  # updating upstream BC (input time series)

            if self.rbc_type == "Dirichlet":  # updating dowsntream BC
                V[-1]   = rbc_n1
            elif self.rbc_type == "Neumann":  # updating dowsntream BC
                V[-1]  =  (1/3)*(2*rbc_n*self.dx) + (4/3)*V[-2] - (1/3)*V[-3]            

            # storing solutions 
            Vout.append(V.copy()) # entire matrix
        
            c_sim = float(V[int(ploc/self.dx)].copy())

            if self.off_typ == "variable" or self.off_typ == "constant" or self.off_typ == "linear":
                c_sim = c_sim + self.oo_int(n*self.dt)
            solsol.append(c_sim)
            
            
            # storing flow direction of last time step 
            if v_current > 0: 
                flow_dir_pre = "down"
            else: 
                flow_dir_pre = "up"
        
        self.modt = modt
        self.cout = solsol
        self.V = V
        self.Vout = Vout
        self.Dhl = Dhl
        
    def print_current_settings(self):
                
        print('dispersivity:', self.a_L )
        print('pcs:', self.pcs )
        print('kappa:', self.kappa0 )
        print('porosity:', self.n )
        print('front_vel:', self.vvs )
        print('conductivity:', self.dds )
        print('real_v:', self.vv_real )
        print('R_therm:',  1+((1-self.n)*self.pcs)/(self.pcw*self.n) )


    def create_matrices(self, v_current, Dh):
        self.seclocnodes = [] # nodes that delineate sections in the coefficient matrix 
        for i in range(0,len(self.seclocs)):
              secbord = self.seclocs[i]      
              self.seclocnodes.append(int(round((secbord-self.seclocs[0])/self.dx,1)))

        # Create matrix coefficients 
        self.lambdas = (Dh*self.dt)/(self.rrs*2*(self.dx**2))  
        self.betas =   (self.kks*self.dt)/(self.rrs*2)              
        
        # alphas depend on transport discretization     
        self.alphas  =((v_current*self.dt)/(self.rrs*4*self.dx))
            # alphas depend on transport discretization     
        if self.advective == "centered":
            self.alphas =((v_current*self.dt)/(self.rrs*4*self.dx))
        elif self.time_dis == "forward" and self.advective == "upwind_1":
            self.alphas =((v_current*self.dt)/(self.rrs*1*self.dx))                
        elif self.time_dis == "LAX"     and self.advective == "upwind_2":
            self.alphas =((v_current*self.dt)/(self.rrs*2*self.dx))

        self.ls_lam = np.mean(self.lambdas); self.ls_alp = np.mean(self.alphas); ls_bet = np.mean(self.betas)
        nx    = len( self.x_dis)  # needed to construct coefficient matrices       

        if self.time_dis == "forward" and self.advective == "centered":
            
            self.A     = diags([-self.ls_lam-self.ls_alp, 1+(2*self.ls_lam)+ls_bet, self.ls_alp-self.ls_lam], [-1, 0, 1],shape=(nx-2, nx-2)).toarray()                    
            self.Bsolo = diags([ self.ls_alp+self.ls_lam, 1-(2*self.ls_lam)-ls_bet, self.ls_lam-self.ls_alp], [-1, 0, 1],shape=(nx-2, nx-2)).toarray()                 
            if self.rbc_type == "Neumann":
                self.A[-1][np.where(self.A[-1] ==1+(2*self.ls_lam)+ls_bet )]=1+(2*self.ls_lam)+ls_bet +(4/3)*(self.ls_alp-self.ls_lam)
                self.A[-1][np.where(self.A[-1] ==(-self.ls_lam-self.ls_alp    ))]=-self.ls_lam - self.ls_alp    -(1/3)*(self.ls_alp-self.ls_lam)



    def update_params(self, params, comps, ini_cons):
        
        self.ini_cons = ini_cons
        
        if "T" in comps and  "EC" in comps:
            start = 1; end = 1
        elif "T" in comps:
            start = 0; end = 0
        elif "EC" in comps:
            start = 1; end = 1            
        

        vvl = [];  #heat transport depends on pcs, n, kappa & v 

        for ii in range(start,int(self.iter+end)):
                vvl.append(params[ii])    

        if self.v_type ==   'variable': 
            self.vvs =vvl
        
        if self.v_type ==   'constant': 
            self.v_const = params[start]  
            vvl =  [self.v_const, self.v_const] 
            self.vvs = vvl
        
        if self.solute == True:
            self.a_L = params[0]

            
            if self.off_typ ==   'linear': 
                self.off_c1 = params[int(self.iter+end)] 
                self.off_c2 = params[int(self.iter+end)+1] 
                self.off =  [self.off_c1, self.off_c2]             
                self.oo_int = interpolate.interp1d(self.oos_t,  self.off)# need to find closes cin value to that point in time t

            if self.off_typ ==   'constant': 
                self.off_c = params[int(self.iter+end)]  
                self.off =  [self.off_c, self.off_c] 
                self.oo_int = interpolate.PchipInterpolator(self.oos_t,  self.off)# need to find closes cin value to that point in time t

            if self.off_typ ==  'variable': 
                ool = [];
                for ii in range(int(self.iter+end), int(self.iter+end)+ int(self.ndays*24/self.h_per_o)+1 ):
                    ool.append(params[ii])
                self.off =ool
                self.oo_int = interpolate.PchipInterpolator(self.oos_t,  self.off)# need to find closes cin value to that point in time t

            self.vv_int = interpolate.PchipInterpolator(self.vvs_t,  (self.vvs)) # interpolation function yields porewater velocity

        if self.solute == False:
            
            self.a_L    = params[-4]
            self.pcs    = params[-3]
            self.n      = params[-1]
            
            
            self.kappa0  = ((self.law**self.n) *(params[-2]**(1- self.n))) #*3600#; print(la)

            
            # vvs denotes thermal front velocity !! Needed for dx estimation 
            self.therm_rfac = (self.pcw/(self.n*self.pcw + (1-self.n)*self.pcs) )*self.n
            
            
            self.vvs = np.array(vvl)*self.therm_rfac # aussumes that porewater velocities are estimated during DREAM
            self.vvt = np.array(vvl)                 # assumes that thermal front velocities are estimated during DREAM
            
            
            # dds denotes D_t, which includes only heat conduction here (mechanical dispersion is added later)
            self.dds = self.kappa0 / (self.n*self.pcw + (1-self.n)*self.pcs)  # [L2/T] here m2/h
            self.vv_real = np.copy(vvl)
            
            #self.vvs = np.array(vvl)
            #self.rrs =  1+     ( (self.n*self.pcw + (1-self.n)*self.pcs) - self.pcw*self.n )/ (self.pcw*self.n)
            #self.dds = self.kappa0 / (self.n*self.pcw)  # [L2/T] here m2/h
        
            # instantiating velocity interpolation 
            if self.joint:        
                self.vv_int = interpolate.PchipInterpolator(self.vvs_t,  (self.vvs)) # interpolation function yields thermal front velocity
                #print(self.vvs)
                #print("here")
            else:
                # thermal front velocities are estimated during dream runs 
                self.vv_int = interpolate.PchipInterpolator(self.vvs_t,  (self.vvt)) # interpolation function yields thermal front velocity
    

        self.ini_cond_make()

    def v_o_int(self):
        
        self.dd_int = interpolate.PchipInterpolator(np.array(self.modt)*self.dt,  np.array(self.Dhl)) # interpolation function yields thermal front velocity

        if self.solute:
            vv_ts = self.vv_int(self.modtime)
            oo_ts = self.oo_int(self.modtime)
        else:
            if self.joint:
                vv_ts = self.vv_int(self.modtime)
            else:
                vv_ts = self.vv_int(self.modtime)/ self.therm_rfac

            oo_ts = self.dd_int(self.modtime)
        return self.modtime, vv_ts, oo_ts
    
    
    def loglikelihood(self, chain, foldname4, start):
        comp_int = interpolate.PchipInterpolator(self.modt, self.cout)

        ttt = self.modtime/self.dt
        cmod = comp_int(ttt)
        # should the likelihood function consider meas. values after an "embargo" period specified by st_t?
        mask_obs = np.where(self.modtime> self.st_t)
        
        
        # assuming that the meas. error is half the accuracy
        if self.comp == "T":  one_std = 0.035 #  according to Munz et al 2011 (acccuracy of 0.07 degC )   + 0.05 # was 0.1 but adjusted degC for ctd, micro and baro divers according to instrument fact sheet
        if self.comp == "EC_T":  one_std = 0.035 # # simulation for temp, jointly with ec         
        if self.comp == "EC": one_std = 0.005 #+ 0.01  # 10 myS/cm 
        
        like =   - 0.5*np.sum(((self.cmea[mask_obs] - cmod[mask_obs]) / one_std)**2)  
        
        if self.solute:
            v_vec = np.copy(self.vvs)
        else:
            # heat as tracer: thermal front velocity differences are used 
            v_vec = np.copy(self.vvt)

        np.mean(abs(np.array(v_vec)))
        relv = np.mean(abs(np.array(v_vec)))
        if self.reldiff:
            vvdlike =   - np.sum(((np.diff(v_vec)/ relv )    )**2)  
        else:
            vvdlike =   - 0.5*np.sum(((np.diff(v_vec))    )**2)  
            
        
        if self.solute == True and self.off_typ != "none":
            if  self.off_typ == "constant" or self.off_typ == "linear":
                oodlike =0
            else:
                if self.reldiff:
                    np.mean(abs(np.array(self.off)))
                    relo = np.mean(abs(np.array(self.off)))
                    oodlike =   - np.sum(((np.diff(self.off)/ relo )    )**2)  
                else:
                    oodlike =   - 0.5*np.sum( (   (np.diff(self.off))    )**2  )  
                    
        else:
            oodlike = 0
        
        L = 0; parfac = 1
        for n in range(chain.npars):
            if chain.uniform[n] == False:
                if start == True:
                    L += (-1./(2.*chain.width[n]**2) * ((chain.mean[n]-chain.current[n])**2.))*parfac
                else:
                    L += (-1./(2.*chain.width[n]**2) * ((chain.mean[n]-chain.proposal[n])**2.))*parfac
        
        like_tot = like + L

        self.like  = copy.deepcopy(like_tot)
        self.cmod  = copy.deepcopy(cmod)

        self.rmse  = copy.deepcopy(np.sqrt(  np.mean((np.array(cmod[mask_obs])-np.array(self.cmea[mask_obs]))**2))  )
        self.sse   = copy.deepcopy(np.sum(  (np.array(cmod[mask_obs])-np.array(self.cmea[mask_obs]))**2 )  )

        self.vvdlike= copy.deepcopy(vvdlike)
        self.oodlike= copy.deepcopy(oodlike)
        self.parlike = copy.deepcopy(L)
        
        return self.like, self.cmod, self.rmse, self.sse, self.vvdlike, self.oodlike , self.parlike , self.cmea, self.lbc  , self.modtime , self.rbc         
    

    def tau_cal(self):
        
        if self.solute:
            vv_ts = self.vv_int(self.modtime)
        else:
            vv_ts = self.vv_int(self.modtime)/self.therm_rfac
        v_rev = vv_ts[::-1]
        t_rev = self.modtime[::-1]
        dt_step = self.modtime[2]- self.modtime[1]
        i = 0; area = 0; t = 0
        tmax = self.modtime[-1]
        times = []; taus = []; taus2 = []
        dt = 0.01
        t = tmax

        for t in self.modtime[self.modtime>=24][::-1] :
            area = 0; area2 = 0
            ttt = t
            while area <= self.ploc:   
                ttt = ttt-dt
                a = abs(self.vv_int(ttt))
                b = abs(self.vv_int(ttt+dt))
                c = (self.vv_int(ttt))
                d = (self.vv_int(ttt+dt))
                if self.solute == False:
                    a = a/ ((self.pcw/(self.n*self.pcw + (1-self.n)*self.pcs) )*self.n)
                    b = b/ ((self.pcw/(self.n*self.pcw + (1-self.n)*self.pcs) )*self.n)
                    c = c/ ((self.pcw/(self.n*self.pcw + (1-self.n)*self.pcs) )*self.n)
                    d = d/ ((self.pcw/(self.n*self.pcw + (1-self.n)*self.pcs) )*self.n)
                               
                if ttt > 0:
                    area += (b+a)*(0.5*dt); area2 += (c+d)*(0.5*dt)
                else:
                    break 
            if area < self.ploc:
                tau = 0
            else:
                tau = t - ttt
            if abs(area2) < self.ploc:
                tau2 = 0
            else:
                tau2 = t - ttt
            times.append(t)
            taus.append(tau); taus2.append(tau); 
        tau_vec = np.array(taus)    
        self.tau_vec2 = np.copy(taus2)
        self.tau_vec = tau_vec
        self.tau_tt = times
        return tau_vec, times, self.tau_vec2


    def dummy_fun(self):
        
        return 0


    def find_dx_Pe(self):
        
        if self.solute == True:
            Dh = 0.7 * self.Dmol + abs(np.array(self.vv_int(self.vvs_t) ) ) *self.a_L

        else:
            Dh = self.dds + abs(np.array(self.vvs))  *  (self.a_L**1)
            Dh = self.dds + abs(np.array(self.vv_int(self.vvs_t) ) ) *  (self.a_L**1)
        
        self.Pe = (self.dx*abs(np.array(self.vv_int(self.vvs_t) ) )) /Dh
        self.Pet = (self.dx*abs(np.array(self.vv_int(self.vvs_t) ) )) /self.dds
        
        max_dx_a = ((2*Dh)/abs(np.array(self.vv_int(self.vvs_t) ) ))

        
        max_dx   = max_dx_a[np.where(max_dx_a == min(max_dx_a))][0]
        istr = str(max_dx)
        digitl = []; zero_count = -1
        for elem in istr:
            if not elem in ('0','.') :
                digitl.append(elem)
            if not elem in ('.') :
                if int(elem) == 0: zero_count +=1 
                else:
                    break        
        twofive = False

        if     9 >= float(digitl[0]) >= 5: digitl[0]='5'; 
        #elif   5 >  float(digitl[0]) >= 4: digitl[0]='4'; 
        elif   5 >  float(digitl[0]) >= 3: digitl[0]='2' ; twofive = True
        elif   3 >  float(digitl[0]) >= 2: digitl[0]='2' ;
        elif   2 >  float(digitl[0]) >= 1: digitl[0]='1' ;
        
        if   twofive == True:
            prop_dx = float('0.' + zero_count*'0' +    digitl[0] + '5' )
        else:
            prop_dx = float('0.' + zero_count*'0' +    digitl[0]  )
            
        #if prop_dx >  0.01: prop_dx = 0.01 # making sure that at dx = at least 1 cm 
        if prop_dx >= 0.01: prop_dx = 0.005 # making sure that at dx = at least 0.5 cm 

        if prop_dx <  0.0005: 
            prop_dx = 0.0005 ; 

        if prop_dx < 0.001: print('adjusted dx =    ',round(prop_dx,5))
    
        return prop_dx

############################################################################################