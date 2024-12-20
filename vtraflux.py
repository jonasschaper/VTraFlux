import ray
import numpy as np
import copy
from scipy.sparse import diags
from scipy import interpolate
from lmfit import minimize, Parameters, report_fit
import time
import matplotlib.pyplot as plt
import os

class vtraflux_model:
    """
    The main vtraflux model class; a one dimensional advection-dispersion 
    model with retardation  

    ...

    Attributes
    ----------
    comp : str
        comp for compound, i.e., the state variable for which transport 
        is to be simulated 


    Methods
    -------
    
    """
    
    def __init__(self, indata,
                       chunk,
                       dt,
                       rbc,
                       seclocs,
                       param_info,
                       comp,
                       ID = None):
        """
        Parameters
        ----------
        indata : dict
            containing the input data (for boundaries and parameter 
                                       estimation)
        chunk : tuple
            specifying the start and end day of the simulation 
        dt : float
            time discretization in hours
        rbc : int, array
            right/lower/downstream bounrdary conditon
        seclocs : list
            depths of the the upstream boundary, calibration depth and
            lower boundary
        param_info : dict
            information on model parameterization
        comp : str
            comp for compound, i.e., the state variable whose transport 
            is to be simulated 
        ID : int
            model ID used for mutli-processing 
        """
        
        self.comp        = comp
        self.tpo         = param_info["tpo"]    # thermal parameter option, eg., "1", "2", "3"
        self.st_t        = param_info["st_t"]   # model time at which second staging post is put
        self.restime_h   = param_info["st_t"]   # time from which measurments are considered in objective function        
        self.T_err       = param_info["T_err"]  # error for temperature 
        self.EC_err      = param_info["EC_err"] # error for EC / solutes 
        self.reldiff     = param_info["reldiff"]  
        self.v_type      = param_info["v_type"]

        self.chunk = chunk 
        self.sel_tvec = np.where((indata[0][1]  <= 24*chunk[1]) & (indata[0][1]  > 24*(chunk[0]-1) ))
        self.modtime = np.array(indata[1][1][self.sel_tvec])-indata[1][1][self.sel_tvec][0]

        self.wv = 0
        self.joint = False 
        self.params = {}


        if comp == "T": 
            self.solute = False
            self.off_typ = "none"
            self.offset = False
        elif comp == "EC":
            self.solute = True
            self.offset = param_info["offset"]
            self.off_typ = param_info["o_type"]
            
        if comp == "EC_T":
            self.joint = True 
            self.solute = False
            self.off_typ = "none"
            self.offset = False

        if len(indata) == 3:
            rbc = indata[-1][-1][self.sel_tvec]
            self.ploc  = indata[1][0] -indata[0][0]
            self.x_max = indata[-1][0]-indata[0][0]
        else:
            self.ploc = indata[-1][0]-indata[0][0]




        self.cmea    = indata[1][2][self.sel_tvec]
        self.cout    = np.zeros((len(indata[0][1]) ,len(seclocs)-1))
        self.ndays   = self.chunk[1]-(self.chunk[0]-1)
        self.iter_count = 0
        self.stagingpost_position = "corner" # "mid 

        if isinstance(rbc, float) == 1:
            self.rbc_type    = "Neumann"
            self.rbc_intvals = rbc* np.ones( len(self.lbc))#*(np.mean(self.lbc)+np.mean(self.cmea))/2 # constant groundwater value 
            self.x_max = (seclocs[-1]-seclocs[0])*2 +.3
        
        elif len(rbc) > 1:
            self.rbc_type    = "Dirichlet"
            if comp == "EC": rbc = copy.deepcopy(rbc[self.sel_tvec])
            
            self.rbc_intvals = copy.deepcopy(rbc)
            if self.solute:
                self.x_max = (seclocs[-1]-seclocs[0])*4 + 0.75 # +1.      
        
        
        # discretization 
        self.dtp      = dt        
        self.nt       = int((indata[0][1][self.sel_tvec][-1]+ (indata[0][1][self.sel_tvec][1]-indata[0][1][self.sel_tvec][0]))/self.dtp)
        self.max_t    = int((indata[0][1][self.sel_tvec][-1]+ (indata[0][1][self.sel_tvec][1]-indata[0][1][self.sel_tvec][0])))
        self.indata   = copy.deepcopy(indata)
        self.lbc      = indata[0][2][self.sel_tvec]
        
        self.dxp = 0.005 if self.solute  else  0.0025
        self.x_dis = np.linspace(0,self.x_max,int(self.x_max/self.dxp)+1)
        
        self.advective   = param_info["advective"]  

        if  param_info["advective"] == "default" and self.solute: 
            self.advective = "centered"
        elif param_info["advective"] == "default" and self.solute == False:
            self.advective = "SUS"
        
        if self.advective == "SUS":
            self.dx = np.copy(self.dxp)

        self.rbc           = rbc
        self.ID            = ID
        self.Dmol          = 0.3e-9*3600 # default vlaue in PHREEQC
        self.seclocs       = seclocs 
        
        # model parameters 
        self.vvs = 0.1
        self.dds = 0.001
        self.kks = 0.0
        self.rrs = 1
        self.pcw = 4.184*1e6   # J / (K  m3)
        self.y   = 0  # zero-order production rate 
        self.pcs = 3.0*1e6
        self.pcb = 3.0*1e6
        self.law = 0.59*3600 # W/3600 m-1 °C-1 thermal conductivity of water at 25 degC
        self.a_L = 0.001
        self.therm_rfac = 0.4

        # staging posts for interpolation values: velocity  
        self.v_per_d = param_info["v_int"] 
        if self.v_type == "variable":
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
        
        
        self.h_per_o = param_info["o_int"]; 
        if self.offset:
            if  self.off_typ == "variable":
                oos_tt = []
                for i in range(int(self.ndays*24/self.h_per_o )+1):
                    oos_tt.append(self.h_per_o*i) # staging posts edges of interval  
                self.oos_t = oos_tt  
            elif  self.off_typ == "constant" or self.off_typ == "linear":
                self.oos_t = [0,self.ndays*24]  
            self.oo_int = interpolate.PchipInterpolator(self.oos_t, 0.0*np.ones(len(self.oos_t)))# need to find closes cin value to that point in time t

        
        self.t_max   = copy.deepcopy(self.indata[0][1][-1])
        self.vv_int  = interpolate.PchipInterpolator(self.vvs_t, 0.1*np.ones(len(self.vvs_t)))# need to find closes cin value to that point in time t
        self.lbc_int = interpolate.PchipInterpolator(self.modtime, self.lbc)
        self.rbc_int = interpolate.PchipInterpolator(self.modtime, self.rbc_intvals)
        
        self.ini_cons = False # linear interpolation
        self.ini_cond_make()

    def ini_cond_make(self):

        if   self.ini_cons:         
            # running model using the first velocity only to get initial condition 
            ini_t = self.st_t
            self.nt = int(ini_t/self.dt)

            # initial condition based on test run 
            self.vv_int = interpolate.PchipInterpolator(self.vvs_t[0:2],  (self.vvs[0:2])) # need to find closes cin value to that point in time t
            self.inc_int      = interpolate.PchipInterpolator(np.array([0,self.ploc,self.x_max]), np.array([self.lbc[0],self.cmea[0], self.rbc_intvals[0]]))# need to find closes cin value to that point in time t
            self.simulation()
            self.inc_int = interpolate.PchipInterpolator(self.x_dis, 1*(np.mean(self.Vout, axis = 0))   )# need to find closes cin value to that point in time t
            self.nt = int((self.indata[0][1][-1]+ (self.indata[0][1][1]-self.indata[0][1][0]))/self.dt)
             
   
        else:
            # simple initial condition via interpolation
            if self.joint == True or  self.off_typ == "none":
                self.inc_int      = interpolate.PchipInterpolator(np.array([0,self.ploc,self.x_max]), np.array([self.lbc[0],self.cmea[0], self.rbc_intvals[0]]))# need to find closes cin value to that point in time t
                self.inc_int_base = interpolate.PchipInterpolator(np.array([0,self.ploc,self.x_max]), np.array([self.lbc[0],self.cmea[0], self.rbc_intvals[0]]))# need to find closes cin value to that point in time t
            else:    
                if self.off_typ == "variable"or self.off_typ == "constant" or self.off_typ == "linear":
                    self.inc_int      = interpolate.PchipInterpolator(np.array([0,self.ploc, self.x_max]), np.array([self.lbc[0] , self.lbc[0]   , self.rbc_intvals[0]  ]))# need to find closes cin value to that point in time t
                    self.inc_int_base = interpolate.PchipInterpolator(np.array([0,self.ploc, self.x_max]), np.array([self.lbc[0] , self.lbc[0]   , self.rbc_intvals[0]  ]))# need to find closes cin value to that point in time t
                   


    def simulation(self):
        cinl = [];Vout  = []; modt = []; solsol = [] ; Dhl =[]; Pel =[];  vvl =[];

        self.zeta    = 0.5
        if self.advective == "centered":
            self.eta = 0.5 # Crank-Nicolson
        if self.advective == "SUS":
            self.eta = 1.0 # fully implicit 
        
        self.find_dxdt()
        
        self.x_dis = np.linspace(0,self.x_max,int(self.x_max/self.dx)+1)
        self.c_in = self.inc_int(self.x_dis)
        v         = copy.deepcopy(self.c_in)                  
        

        for n in range(0,int(self.max_t/self.dt)-1):  
            modt.append(n+1)            
            v_current = float((self.vv_int(n*self.dt)))
            
            if self.solute:
                Dh = 0.7 * self.Dmol + abs(v_current)*self.a_L
            else: # v_current is thermal front velocity when heat as tracer
                Dh = self.dds + abs(v_current)*(self.a_L**1) #rau 2014 heat review
            Dhl.append(Dh); vvl.append(v_current)
            self.create_matrices(v_current, Dh,n)

            lbc_n  = float(self.lbc_int(n*self.dt))
            rbc_n  = float(self.rbc_int(n*self.dt))
            lbc_n1 = float(self.lbc_int( (n+1)*self.dt  ))
            rbc_n1 = float(self.rbc_int( (n+1)*self.dt ))
                        
            b  = np.dot(self.Bsolo,v) 
            
            b[0] = lbc_n1 # upper BC
            
            if self.rbc_type == "Dirichlet":   # lower BC
                b[-1] = rbc_n1
            elif self.rbc_type == "Neumann": 
                b[-1] = (2*self.dx)*rbc_n  
    
            b += self.y*self.dt # adding a zero order production rate (future use)
            
            v = np.linalg.solve(self.A,b)# v stores solution of n+1 timestep 
            cinl.append(lbc_n1) 

            Vout.append(v.copy())
        
            c_sim = float(v[int(self.ploc/self.dx)].copy())
            if self.offset:
                if self.off_typ in ["variable", "constant", "linear"]:
                    c_sim = c_sim + self.oo_int(n*self.dt)
            solsol.append(c_sim);Pel.append((self.dx*abs(v_current))/Dh)  
            
        self.modt = modt
        self.cout = solsol
        self.V = v
        self.Vout = Vout
        self.Dhl = Dhl; self.vvl = vvl
        self.Pel = Pel
        
    
    def create_matrices(self, v_current, Dh,n):        
        nx = len(self.x_dis)
        self.lambdas = (Dh*self.dt)/(self.rrs*(self.dx**2))  
        self.betas =   (self.kks*self.dt)/(self.rrs*2)        
        if self.advective == "centered":
            self.alphas =((v_current*self.dt)/(self.rrs*2*self.dx))
            self.ldA = -self.lambdas*self.eta-self.zeta*self.alphas
            self.mdA = 1+(2*self.lambdas*self.eta)+self.betas
            self.udA = -self.lambdas*self.eta+self.zeta*self.alphas
            self.A   = diags([self.ldA, self.mdA, self.udA], [-1, 0, 1],
                           shape=(nx-0,nx-0)).toarray()                    
            
            self.ldB = self.lambdas*(1-self.eta)+(1-self.zeta)*self.alphas
            self.mdB = 1-(2*self.lambdas*(1-self.eta))-self.betas
            self.udB = self.lambdas*(1-self.eta)-(1-self.zeta)*self.alphas
            self.Bsolo = diags([self.ldB, self.mdB, self.udB],[-1, 0, 1],
                               shape=(nx-0, nx-0)).toarray()                 
            
        if self.advective == "SUS":
            Pe = (self.dx*abs(v_current))/Dh
            self.gamma = abs(np.cosh(Pe/2)/np.sinh(Pe/2)-2/Pe) if Pe < 0.1 else abs(Pe/6)
            #self.gamma = 1 / np.tanh(Pe / 2) - 2 / Pe if Pe != 0 else Pe / 6  
            self.alphahat = ((v_current*self.dt)/(self.rrs*1*self.dx*2))                

            self.ldA = -self.lambdas*self.eta - self.zeta*self.alphahat*(1-self.gamma)    
            self.mdA = ( 1+(2*self.lambdas*self.eta)+self.betas - 
            (self.zeta*2*self.alphahat*(self.gamma)) )
            self.udA = -self.lambdas*self.eta + self.zeta*self.alphahat*(1+self.gamma)
                                          
            self.ldB = self.lambdas*(1-self.eta) + (1-self.zeta)*self.alphahat*(1-self.gamma)        
            self.mdB = (1-(2*self.lambdas*(1-self.eta))-self.betas + 
            ((1-self.zeta)*2*self.alphahat*(self.gamma)) )
            self.udB = self.lambdas*(1-self.eta) - (1-self.zeta)*self.alphahat*(1+self.gamma)

            self.Bsolo=diags([self.ldB,self.mdB,self.udB],[-1,0,1],
                             shape=(nx-0,nx-0)).toarray()   
            self.A   = diags([self.ldA, self.mdA, self.udA], [-1,0,1],
                           shape=(nx-0,nx-0)).toarray() 

        # lower BC, Dirichlet or Neumann
        if self.rbc_type == "Dirichlet":  
            self.A[-1,-1]= 1
            self.A[-1,-2]= 0
            self.Bsolo[-1,-1]= 1
            self.Bsolo[-1,-2]= 0
        elif self.rbc_type == "Neumann":
            self.A[-1,-1] = 3  
            self.A[-1,-2] = -4
            self.A[-1,-3] = 1
            self.Bsolo[-1,-1] = 3 
            self.Bsolo[-1,-2] = -4
            self.Bsolo[-1,-3] = 1
            
        # upper BC, always Dirichlet
        self.A[0,0]= 1
        self.A[0,1]= 0
        self.Bsolo[0,0]= 1
        self.Bsolo[0,1]= 0

            
    def find_dxdt(self):

        prop_dx =np.copy(self.dxp)  
        prop_dt =np.copy(self.dtp)  
        dt_mx = 0.2 
        
        Dh = self.dds + abs(np.array(self.vv_int(self.vvs_t) ) ) *  (self.a_L**1)
        vvs = self.vv_int(self.vvs_t)
                
        # set dx
        if self.advective == "centered":
            dx_pec =  min(Dh)*2/max(abs(vvs)) # global, worst-case Pe
            dx     = min(prop_dx, dx_pec) # based on Peclet number
            if dx <  0.001: dx = 0.001;    
            self.dx = dx
        else:
            self.dx = prop_dx

        # set dt
        dt_cfl = self.CFL(dt_mx = dt_mx); # CFL criterion 
        dt_neumann = (self.dx ** 2) / (3 * max(Dh))        


        if self.zeta < 1 and self.eta == 1.0: 
            # implicit in diffusion 
            self.dt = dt_cfl
        elif self.zeta == 0.5 and self.eta == 0.5:
            # crank nicolson
            Pe_min = dx*min(abs(vvs))/ max(Dh) # global, worst-case Pe        
            if Pe_min > 0.1:         
                self.dt = min( dt_cfl, prop_dt)
            else:
                self.dt = min( dt_cfl, dt_neumann, prop_dt)
        else:
            self.dt = min( dt_neumann,  dt_cfl, prop_dt)
            
        if self.dt < 0.025: 
            self.dt = dt_cfl
            self.eta = 1.0

    def CFL(self, dt_mx):
        # CFL / courant: find min of dx/abs(v) 
        vvs = self.vv_int(self.vvs_t)
        dt_cfl = 1* self.dxp/max(abs(vvs))

        if  dt_cfl > dt_mx:  dt_cfl = dt_mx
        return dt_cfl

    def update_params(self, params, comps, ini_cons):
        
        self.ini_cons = ini_cons
        
        if "T" in comps and  "EC" in comps:
            start = 1; end = 1
        elif "T" in comps:
            start = 0; end = 0
        elif "EC" in comps:
            start = 1; end = 1            
        

        vvl = []; 

        for ii in range(start,int(self.iter+end)):
                vvl.append(params[ii])    

        if self.v_type ==   'variable': 
            self.vvs =vvl
        
        if self.v_type ==   'constant': 
            self.v_const = params[start]  
            vvl =  [self.v_const, self.v_const] 
            self.vvs = vvl
        
        if self.solute:
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
            
            self.a_L    = params[-1]
            
            if self.tpo == "TPO_01":
                self.pcs    = params[-4]
                self.n      = params[-2]
                self.kappa0  = ((self.law**self.n) *(params[-3]**(1- self.n))) #*3600#; print(la)
    
                # vvs denotes thermal front velocity; Needed for dx estimation 
                self.therm_rfac = (self.pcw/(self.n*self.pcw + (1-self.n)*self.pcs) )*self.n

                # dds denotes thermal diffusivity, which includes only heat conduction here (mechanical dispersion is added later)
                self.dds = self.kappa0 / (self.n*self.pcw + (1-self.n)*self.pcs)  # [L2/T] here m2/h
                self.vv_real = np.copy(vvl)
            
            
            if self.tpo == "TPO_02":
                
                self.pcb     = params[-4]
                self.n       = params[-2]
                self.kappa0  = params[-3] 

                self.therm_rfac = (self.pcw/( self.pcb  ) )*self.n
                self.dds = self.kappa0 / (self.pcb)  # [L2/T] here m2/h

            if self.tpo == "TPO_03":
                self.therm_rfac = params[-2]
                self.dds        = params[-3]
                                                    
            # instantiating velocity interpolation 
            if self.joint: # heat is simulated together with solute transport    
                self.vvt = np.array(vvl)*self.therm_rfac # # conversion to thermal front velocity ...
                self.vvs = np.array(vvl)   # as porewater velocities are estimated during param. est.              

            else:
                # in heat only runs, thermal front velocities are estimated 
                self.vvt = np.array(vvl) # assumes that thermal front velocities are estimated during param. est.
                self.vvs = np.array(vvl) / self.therm_rfac               

                # thermal front velocities are estimated during dream runs 
            self.vv_int = interpolate.PchipInterpolator(self.vvs_t,  (self.vvt)) # interpolation function yields thermal front velocity
    

        self.ini_cond_make()

    def v_o_int(self):
        
        self.dd_int = interpolate.PchipInterpolator(np.array(self.modt)*self.dt,  np.array(self.Dhl)) # interpolation function yields thermal front velocity

        if self.solute:
            vv_ts = self.vv_int(self.modtime)
            oo_ts = self.oo_int(self.modtime)
        else:
            # heat transport, back-convert from thermal front to pw-vel
            if self.joint:
                vv_ts = self.vv_int(self.modtime)/ self.therm_rfac
            else:
                vv_ts = self.vv_int(self.modtime)/ self.therm_rfac
            oo_ts = self.dd_int(self.modtime)
            
            
        return self.modtime, vv_ts, oo_ts
    
    
    def loglikelihood(self, chain, foldname4, start):
        comp_int = interpolate.PchipInterpolator(self.modt, self.cout)

        ttt = self.modtime/self.dt
        cmod = comp_int(ttt)
        #mask_obs = np.where(self.modtime > self.st_t)
        
        obs_24 = np.where(self.modtime <= self.st_t)
        obs    = np.where(self.modtime >  self.st_t)

        if self.comp == "T":     one_std = self.T_err  #  according to Munz et al 2011 (acccuracy of 0.07 degC )  
        if self.comp == "EC_T":  one_std = self.T_err  #  simulation of heat, jointly with ec         
        if self.comp == "EC":    one_std = self.EC_err
                
        like =   - (0.5*np.sum(((self.cmea[obs_24] - cmod[obs_24]) / one_std)**2)  )*0.25
        like -=    0.5*np.sum(((self.cmea[obs] - cmod[obs]) / one_std)**2)  

        if self.solute:
            v_vec = np.copy(self.vvs)
        else:
            # heat as tracer:
            if self.joint:
                #porewater vel is used when joint simulation with solute tracer
                v_vec = np.copy(self.vvs)
            else:
                # thermal front velocity differences are used when heat is only tracer
                v_vec = np.copy(self.vvt) 
        
        strvel = int(self.st_t/self.v_per_d+1)
        if self.st_t == 0: strvel:0
        
        relv = np.mean(abs(np.array(v_vec[strvel:])))
        if self.reldiff:
            vvdlike =   - np.sum(((np.diff(v_vec[strvel:])/ relv )    )**2)  
        else:
            vvdlike =   - np.sum(((np.diff(v_vec[strvel:]))    )**2)  
            
        
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

        self.rmse  = copy.deepcopy(np.sqrt(  np.mean((np.array(cmod[obs])-np.array(self.cmea[obs]))**2))  )
        self.sse   = copy.deepcopy(np.sum(  (np.array(cmod[obs])-np.array(self.cmea[obs]))**2 )  )

        self.vvdlike= copy.deepcopy(vvdlike)
        self.oodlike= copy.deepcopy(oodlike)
        self.parlike = copy.deepcopy(L)
        
        return self.like, self.cmod, self.rmse, self.sse, self.vvdlike, self.oodlike , self.parlike , self.cmea, self.lbc  , self.modtime , self.rbc_intvals         
    

    def tau_cal(self):
        
        tmax = self.modtime[-1]
        times = []; taus = []; taus2 = []
        dt = 0.01
        t = tmax
        tau = 0; #tau2 = 2
        for t in self.modtime[self.modtime>=24][::-1] :
            area = 0; area2 = 0
            ttt = t
            
            if self.solute == True or self.joint == True:

                while area <= self.ploc:   
                    ttt = ttt-dt
                    a = abs(self.vv_int(ttt))
                    b = abs(self.vv_int(ttt+dt))
                    c = (self.vv_int(ttt))
                    d = (self.vv_int(ttt+dt))

    
                    if ttt > 0:
                        area += (b+a)*(0.5*dt);# area2 += (c+d)*(0.5*dt)
                        print(area2)
                    else:
                        break 
            
            if self.solute == False and self.joint == False:
                
                while area2 <= self.ploc:   
                    ttt = ttt-dt
                    a = abs(self.vv_int(ttt))
                    b = abs(self.vv_int(ttt+dt))
                    c = (self.vv_int(ttt))
                    d = (self.vv_int(ttt+dt))
                    
                    a = a/self.therm_rfac
                    b = b/self.therm_rfac
                    c = c/self.therm_rfac
                    d = d/self.therm_rfac
                    
                    if c < 0: c = 0; area = 0
                    if d < 0: d = 0; area = 0
    
                    if ttt > 0:
                        area += (b+a)*(0.5*dt); area2 += (c+d)*(0.5*dt)
                    else:
                        break 

            if area < self.ploc:
                tau = 0
            else:
                tau = t - ttt
            #if abs(area2) < self.ploc:
            #    tau2 = 0
            #else:
            #    tau2 = t - ttt
            times.append(t)
            taus.append(tau); taus2.append(tau); 
            
            
        tau_vec = np.array(taus)    
        self.tau_vec2 = np.copy(taus2)
        self.tau_vec = tau_vec
        self.tau_tt = times
        return tau_vec, times, self.tau_vec2


    def dummy_fun(self):
        return 0

    def iter_fun(self, params, iteriter, res):
        self.iter_count += 1
        print(self.iter_count)
        like2, cmod2, rmse2, sse, vvdlike, oodlike, la, lu, l = self.loglikelihood_02()
        

    def vel2paramdict(self, chain, start, end, params):
        
        vvl = [];  #heat transport depends on pcs, n, kappa & v 
        if self.v_type ==   'variable': 
            for ii in range(start,int(self.iter+end)):
                    vvl.append(chain.current[ii])    
                    params.add("v"+str(ii), value= chain.current[ii],  min = chain.mean[ii]-chain.width[ii], max = chain.mean[ii] + chain.width[ii])
            self.vvs =vvl
        
        if self.v_type ==   'constant': 
            self.v_const = chain.current[start]  
            vvl =  [self.v_const, self.v_const] 
            self.vvs = vvl
            params.add("v_const", value = chain.current[start],  min = chain.mean[start]-chain.width[start], max = chain.mean[start] + chain.width[start])
    
        return params 
    
    def thermal2paramdict(self, chain, start, end, params):
        
        varyvary = True
        
        if self.tpo == "TPO_01":
            self.n      = chain.current[-2]
            self.kappa0  = ((self.law**self.n) *(chain.current[-3]**(1- self.n))) #*3600#; print(la)
            self.pcs    = chain.current[-4]
            params.add("pcs",    value= chain.current[-4],  min = chain.mean[-4]-chain.width[-4], max = chain.mean[-4] + chain.width[-4], vary = varyvary)
            params.add("kappa0", value= chain.current[-3],  min = chain.mean[-3]-chain.width[-3], max = chain.mean[-3] + chain.width[-3], vary = varyvary)
            params.add("n",      value= chain.current[-2],  min = chain.mean[-2]-chain.width[-2], max = chain.mean[-2] + chain.width[-2], vary = varyvary)

            # vvs denotes thermal front velocity; Needed for dx estimation 
            self.therm_rfac = (self.pcw/(self.n*self.pcw + (1-self.n)*self.pcs) )*self.n
                        
            # dds denotes thermal diffusivity, which includes only heat conduction here (mechanical dispersion is added later)
            self.dds = self.kappa0 / (self.n*self.pcw + (1-self.n)*self.pcs)  # [L2/T] here m2/h
        
        
        if self.tpo == "TPO_02":
            
            self.pcb     = chain.current[-4]
            self.n       = chain.current[-2]
            self.kappa0  = chain.current[-3] 
            params.add("pcb",    value= chain.current[-4],  min = chain.mean[-4]-chain.width[-4], max = chain.mean[-4] + chain.width[-4], vary = varyvary)
            params.add("kappa0", value= chain.current[-3],  min = chain.mean[-3]-chain.width[-3], max = chain.mean[-3] + chain.width[-3], vary = varyvary)
            params.add("n",      value= chain.current[-2],  min = chain.mean[-2]-chain.width[-2], max = chain.mean[-2] + chain.width[-2], vary = varyvary)

            self.therm_rfac = (self.pcw/( self.pcb  ) )*self.n
            self.dds = self.kappa0 / (self.pcb)  # [L2/T] here m2/h

        if self.tpo == "TPO_03":
            
            self.therm_rfac = chain.current[-2]
            self.dds        = chain.current[-3]
            
            params.add("dds",    value= chain.current[-4],  min = chain.mean[-4]-chain.width[-4], max = chain.mean[-4] + chain.width[-4], vary = varyvary)
            params.add("therm_rfac", value= chain.current[-3],  min = chain.mean[-3]-chain.width[-3], max = chain.mean[-3] + chain.width[-3], vary = varyvary)

        params.add("beta0",    value= chain.current[-1],  min = chain.mean[-1]-chain.width[-1], max = chain.mean[-1] + chain.width[-1], vary = varyvary)            
 
        return params
        
    def off2paramdict(self,chain, start, end, params):
        

        if  self.off_typ == "variable":
            oos_tt = []
            for i in range(int(self.ndays*24/self.h_per_o )+1):
                oos_tt.append(self.h_per_o*i) # staging posts edges of interval  
            self.oos_t = oos_tt  
        elif  self.off_typ == "constant" or self.off_typ == "linear":
            self.oos_t = [0,self.ndays*24]  
                 
        
        if self.off_typ ==   'linear': 
            self.off_c1 = chain.current[int(self.iter+end)] 
            self.off_c2 = chain.current[int(self.iter+end)+1] 
            self.off =  [self.off_c1, self.off_c2]             
            self.oo_int = interpolate.interp1d(self.oos_t,  self.off)# need to find closes cin value to that point in time t
            params.add("off_c1",    value= chain.current[int(self.iter+end)],  min = chain.mean[int(self.iter+end)]-chain.width[int(self.iter+end)], max = chain.mean[int(self.iter+end)] + chain.width[int(self.iter+end)], vary = True)
            params.add("off_c2",    value= chain.current[int(self.iter+end)+1],  min = chain.mean[int(self.iter+end)+1]-chain.width[int(self.iter+end)+1], max = chain.mean[int(self.iter+end)+1] + chain.width[int(self.iter+end)+1], vary = True)

        if self.off_typ ==   'constant': 
            self.off_c = chain.current[int(self.iter+end)]  
            
            params.add("off_c",    value= chain.current[int(self.iter+end)],  min = chain.mean[int(self.iter+end)]-chain.width[int(self.iter+end)], max = chain.mean[int(self.iter+end)] + chain.width[int(self.iter+end)], vary = True)

            self.off =  [self.off_c, self.off_c] 
            self.oo_int = interpolate.PchipInterpolator(self.oos_t,  self.off)# need to find closes cin value to that point in time t

        if self.off_typ ==  'variable': 
            ool = [];
            for ii in range( int(self.ndays*24/self.h_per_o)+1 ):
                ool.append(chain.current[ii])
                params.add("o"+str(ii), value= chain.mean[ii],  min = chain.mean[ii]-chain.width[ii], max = chain.mean[ii] + chain.width[ii])

            self.off =ool
            self.oo_int = interpolate.PchipInterpolator(self.oos_t,  self.off)# need to find closes cin value to that point in time t

        return params
        
    def chain2paramdict(self, params, chain, comps, w):
        self.ini_cons = False
        
        if "T" in comps and  "EC" in comps:
            start = 1; end = 1
            # solute False run to simulate heat transport with joint EC inversion 
            params.add("a_L",    value= chain.current[0],  min = chain.mean[0]-chain.width[0], max = chain.mean[0] + chain.width[0], vary = True)
            params = self.vel2paramdict(chain, start, end, params)
            params = self.off2paramdict(chain, start, end, params)
            params = self.thermal2paramdict(chain, start, end, params)

        elif "T" in comps:
            start = 0; end = 0
            params = self.vel2paramdict(chain, start, end, params)
            params = self.thermal2paramdict(chain, start, end, params)
            
        elif "EC" in comps:
            start = 1; end = 1            
            params.add("a_L",    value= chain.current[0],  min = chain.mean[0]-chain.width[0], max = chain.mean[0] + chain.width[0], vary = True)
            params = self.vel2paramdict(chain, start, end, params)
            params = self.off2paramdict(chain, start, end, params)
           
        return params

    def paramdict2model(self, params):

        
        current = []
        for ii in range(len(list(params.values()))):
            current.append(list(params.values())[ii].value)
        #print(current)
        
        self.update_params( current, self.comp, ini_cons = False)
        
        
        return current

    def params2file(self, params, dirName):
        
        self.param2model(params)
        self.simulation() 
        like2, cmod2, rmse2, sse, vvdlike, oodlike, parlike, la, lu = self.loglikelihood_02()

        f = open(dirName +  '/'+"w" + str(int(self.wv))+'.dat','w')
        for para in params.keys():
            f.write('%s %g \n' % (para, params[para].value))
        f.write('%s %g \n' % ("x2", np.log10(vvdlike)))
        f.write('%s %g \n' % ("e2", np.log10(sse)))
        f.close()
        return vvdlike, sse
        
    
    def getresiduals(self):
        
        comp_int = interpolate.PchipInterpolator(self.modt, self.cout)
        
        ttt = self.modtime/self.dt
        self.cmod = comp_int(ttt)
        mask_obs = np.where(self.modtime> self.restime_h)
        res = (self.cmea[mask_obs] - self.cmod[mask_obs])
        

        self.residuals = res# **2
        return res



    def objective(self, params):
        
        self.paramdict2model(params)
        self.simulation()            
        
        comp_int = interpolate.PchipInterpolator(self.modt, self.cout)
        
        ttt = self.modtime/self.dt
        self.cmod = comp_int(ttt)
        mask_obs = np.where(self.modtime> self.restime_h)
        res = (self.cmea[mask_obs] - self.cmod[mask_obs])
        

        if self.wv != 0:
            if self.solute == False:
                res = np.append(res, (np.diff(self.vvs)**1) *self.wv  )
            if self.solute == True:
                res = np.append(res, (np.diff(self.vvs)**1) *self.wv)
                res = np.append(res, (np.diff(self.off)**1)*self.wo)
    

        self.residuals = res# **2
        return res
    
    def model_run(self, method): 

      
        mini_res = minimize(self.objective, self.params, method=method, kws = None, iter_cb = self.iter_fun )        

        self.mini_res = mini_res
        self.paramdict2model(mini_res.params)
        
        return mini_res

    
    def currents_plot(self, chain, dirName):
        
        self.simulation()            
        
        like, cmod, rmse, sse, vvdlike, oodlike , parlike , cmea, lbc  , modtime , lbc  =self.loglikelihood(chain, dirName, start = True)
   
        if "T" in self.comp : 
            plt.plot(self.modtime/24, self.lbc, label="upper boundary" , color = '#808080')
            plt.plot(self.modtime/24, self.rbc, label="lower boundary" , color = '#d95319')
            plt.scatter(self.modtime/24, self.cmea,  s = 1, label="meas." , marker = ",")   
            plt.plot(self.modtime/24, cmod, label="mod.", color = '#edb120' )
            plt.xlabel('days')
            plt.ylabel('Temp (°C)')
            plt.legend(loc="upper left")
            plt.plot(figsize=(15,6))
            plt.rcParams['savefig.dpi']=300
            plt.savefig(            dirName+'/'+ "T"+ ".png")
            plt.show()

        
        if "EC" in self.comp: 
            #self.off_typ = param_info["o_type"]
            plt.plot(self.modtime/24, self.lbc, label="upper boundary" , color = '#808080')
            plt.scatter(self.modtime/24, self.cmea,  s = 1, label="meas." , marker = ",")   
            plt.plot(self.modtime/24, cmod, label="mod.", color = '#edb120' )
            plt.xlabel('days')
            plt.ylabel(r'EC [$m  S ~ cm^{-1}$]')
            plt.legend(loc="upper left")
            plt.plot(figsize=(15,6))
            plt.rcParams['savefig.dpi']=300
            plt.savefig(            dirName+'/'+"EC"+ ".png")
            plt.show()
            
            
    def print_current_settings(self):
                
        print('dispersivity:', self.a_L )
        print('pcs:', self.pcs )
        print('kappa:', self.kappa0 )
        print('porosity:', self.n )
        print('front_vel:', self.vvs )
        print('conductivity:', self.dds )
        print('real_v:', self.vv_real )
        print('R_therm:',  1+((1-self.n)*self.pcs)/(self.pcw*self.n) )

        
############################################################################################


