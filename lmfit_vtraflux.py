import ray
from lmfit import minimize, Parameters, fit_report
import numpy as np
import copy
import pandas as pd
import os
import random 

class vtraflux_lmfit:
    """
    vtraflux_lmfit class to estimate parameters of VTraFlux using lmfit 
    
    """
    
    def __init__(self, dream,
                       method,
                       models,
                       comps,
                       ID = None):
        
        self.dream       = dream
        self.comp         = comps["comps"] # state variable to be simulated
        self.comps         = comps # thermal parameter option, eg., "1", "2", "3"
        self.reldiff = True
        self.models =  models[0 ]

        if "T" in self.comp:
            self.T_model = copy.deepcopy(self.models["T"])
        if "EC" in self.comp: 
            self.ec_model = copy.deepcopy(self.models["EC"])
        

        model = models[0 ]
        self.dirName      = model["dirName"]
        self.w = model["wv"]

        
        self.method = method
        # randomly select a chain of which the current parameters will be used as starting values 
        self.chain =  self.dream.chains[random.randint(0, self.dream.nc-1)]

        params   = Parameters()# params['v'].value
        
        if "T" in self.comp and "EC" in self.comp:
            self.T_model.off_typ = self.ec_model.off_typ

            self.params   = self.T_model.chain2paramdict(params, self.chain, "EC_T", self.w)   
            
        elif "T" in self.comp:
            self.params   = self.T_model.chain2paramdict(params, self.chain, "T", self.w)
        elif "EC" in self.comp: 
            self.params   = self.ec_model.chain2paramdict(params, self.chain, "EC", self.w)
        
        self.npars = len(list(self.params .values()))
        self.vv_res = 0
        self.oo_res = 0


    def model_run(self): 

        params = copy.deepcopy(self.params)

        feval_count = 0
        if self.npars > 20: 
            maxfev = 1000
        else: 
            maxfev = 500       
        
        self.mini_res = minimize(self.objective, params, method=self.method, iter_cb = self.iter_fun, max_nfev = maxfev )        

        while self.mini_res.nfev >= maxfev:
            print("Maximum number of function evaluations reached. Restarting with new initial values.")
            self.chain =  copy.deepcopy(self.dream.chains[random.randint(0, self.dream.nc-1)])
            

            if "T" in self.comp and "EC" in self.comp:
                self.params   = self.T_model.chain2paramdict(params, self.chain, "EC_T", self.w)            
            elif "T" in self.comp:
                self.params   = self.T_model.chain2paramdict(params, self.chain, "T", self.w)
            elif "EC" in self.comp: 
                self.params   = self.ec_model.chain2paramdict(params, self.chain, "EC", self.w)

            params = copy.deepcopy(self.params)
            self.mini_res = minimize(self.objective, params, method=self.method, iter_cb = self.iter_fun, max_nfev = maxfev )        
            feval_count += 1
        
        return self.mini_res
    
    
    def print_results(self):
        
        # Save the fit result to a text file
        with open(self.dirName +'/' 'fit_report.txt', 'w') as f:
            f.write(fit_report(self.mini_res))
                
        if "EC" in self.comp: 

            current  = self.ec_model.paramdict2model( self.mini_res.params)
            self.ec_model.simulation()            

            #self.ec_model.currents_plot(self.chain, self.dirName)


            like_solo, cmod, rmse, sse, vvdlike, oodlike , parlike , cmea, lbc  , modtime , rbc =self.ec_model.loglikelihood(self.chain , self.dirName, start = True)
            tot_like    =  like_solo + vvdlike * self.w + oodlike* self.w  
        
            modt, vvs, oos = self.ec_model.v_o_int()
            
            ecmoddf = pd.DataFrame(modtime)
            ecmoddf['cmea'] = cmea
            ecmoddf['c_in'] =lbc
            ecmoddf['c_ou'] =lbc
            ecmoddf.to_csv(os.path.join(self.dirName) +  '/'+ "ec_cmod" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')
            
            
            tau_vec, timev, tau_vec2 = self.ec_model.tau_cal()

            
            f = open(os.path.join(self.dirName) +  '/'+ "ec_tau_ml" + '.txt','w')
            dim = np.shape(tau_vec)
            for k in range(dim[0]):
                f.write('%g %g \n' %  (timev[k], tau_vec[k]) )        
            f.write('\n')
            f.close()

            f = open(os.path.join(self.dirName) +  '/'+ "ec_v" + '.txt','w')
            dim = np.shape(modt)
            for k in range(dim[0]):
                f.write('%g %g \n' %  (modt[k], vvs[k]) )        
            f.write('\n')
            f.close()
            
            f = open(os.path.join(self.dirName) +  '/'+ "ec_d" + '.txt','w')
            dim = np.shape(modt)
            for k in range(dim[0]):
                f.write('%g %s \n' %  (modt[k],   str('{:04.5f}'.format(oos[k]) )     ) )        
            f.write('\n')
            f.close()
            
            # save estimated max like params 
            postpars = np.append(np.copy(tot_like),  np.append(np.copy(current), np.append(np.copy(rmse), np.append(np.copy(sse), np.append(np.copy(like_solo), np.append(np.copy(oodlike), np.append(np.copy(vvdlike), np.copy(1) )))))))

            f = open(self.dirName +  '/'+'postpars.dat','w')
            dim = np.shape(postpars)
            for k in range(dim[0]):
                f.write('%g ' % postpars[k])
            f.write('\n')

            f = open(self.dirName +  '/'+'current.dat','w')
            dim = np.shape(current)
            f.write('%g ' % np.copy(tot_like))
            for j in range(dim[0]):
                f.write('%g ' % current[j])
            f.write('\n')
            f.close()


        if "T" in self.comp: 
            
            current  = self.T_model.paramdict2model( self.mini_res.params)
            self.T_model.simulation()            
            
            #self.T_model.currents_plot(self.chain, self.dirName)

            like_solo, cmod, rmse, sse, vvdlike, oodlike , parlike , cmea, lbc  , modtime , rbc = self.T_model.loglikelihood(self.chain , self.dirName, start = True)
    
            
            tot_like    =  like_solo + vvdlike * self.w + oodlike* self.w  
    
            modt, vvs, oos = self.T_model.v_o_int()

            Tcmoddf = pd.DataFrame(modtime)
            Tcmoddf['cmea'] = cmea
            Tcmoddf['c_in'] =lbc
            Tcmoddf['c_ou'] =rbc
            Tcmoddf.to_csv(os.path.join(self.dirName) +  '/'+ "T_cmod" + '.txt', header=None, index=None, sep=' ', mode='w', float_format='%.4f')

            #tau_vec, timev, tau_vec2 = self.T_model.tau_cal()

            
            #f = open(os.path.join(self.dirName) +  '/'+ "ec_tau_ml" + '.txt','w')
            #dim = np.shape(tau_vec)
            #for k in range(dim[0]):
            #    f.write('%g %g \n' %  (timev[k], tau_vec[k]) )        
            #f.write('\n')
            #f.close()

            f = open(os.path.join(self.dirName) +  '/'+ "T_v" + '.txt','w')
            dim = np.shape(modt)
            for k in range(dim[0]):
                f.write('%g %g \n' %  (modt[k], vvs[k]) )        
            f.write('\n')
            f.close()

            f = open(os.path.join(self.dirName) +  '/'+ "T_d" + '.txt','w')
            dim = np.shape(modt)
            for k in range(dim[0]):
                f.write('%g %s \n' %  (modt[k],   str('{:04.5f}'.format(oos[k]) )     ) )        
            f.write('\n')
            f.close()

            # save estimated max like params 
            postpars = np.append(np.copy(tot_like),  np.append(np.copy(current), np.append(np.copy(rmse), np.append(np.copy(sse), np.append(np.copy(like_solo), np.append(np.copy(oodlike), np.append(np.copy(vvdlike), np.copy(1) )))))))
            
            f = open(self.dirName +  '/'+'postpars.dat','w')
            dim = np.shape(postpars)
            for k in range(dim[0]):
                f.write('%g ' % postpars[k])
            f.write('\n')
    
            f = open(self.dirName +  '/'+'current.dat','w')
            dim = np.shape(current)
            f.write('%g ' % np.copy(tot_like))
            for j in range(dim[0]):
                f.write('%g ' % current[j])
            f.write('\n')
            f.close()
            
            

    
    def objective(self,  params):
        
        w          = copy.deepcopy(self.w)
        comp       = copy.deepcopy(self.comp)


        if "T" in comp:
            T_model    = copy.deepcopy(self.T_model)
            
            T_model.paramdict2model(params)
            T_model.simulation()            
            T_res = np.copy(T_model.getresiduals() )       
            
            # thermal front velocity is used 
            v_vec = np.copy(T_model.vvt)
            
            if T_model.joint:
                v_vec = np.copy(T_model.vvs)

            
            #res = np.append(T_res, (np.diff(T_model.vvs)**1) *  w) #  T_model.wv  )
            relv = np.mean(abs(np.array(v_vec)))
            if self.reldiff:
                self.vv_res = np.copy( ((np.diff(v_vec)/ relv )    )**2 ) 
            else:
                self.vv_res =  np.copy( ((np.diff(v_vec))    )**2  )
                
        if "EC" in comp: 
            ec_model   = copy.deepcopy(self.ec_model)

            ec_model.paramdict2model(params)
            ec_model.simulation()            
            ec_res = np.copy(ec_model.getresiduals())  # residual vector of model fit 
            
     
            v_vec = np.copy(ec_model.vvs)
            np.mean(abs(np.array(v_vec)))
            relv = np.mean(abs(np.array(v_vec)))
            if self.reldiff:
                self.vv_res =  np.copy(((np.diff(v_vec)/ relv )    )**2  )
            else:
                self.vv_res =  np.copy( ((np.diff(v_vec))    )**2  )
                
            
            if ec_model.solute == True and ec_model.off_typ != "none":
                if  ec_model.off_typ == "constant" or ec_model.off_typ == "linear":
                    self.oo_res
                else:
                    if self.reldiff:
                        np.mean(abs(np.array(ec_model.off)))
                        relo = np.mean(abs(np.array(ec_model.off)))
                        self.oo_res =   np.copy(((np.diff(ec_model.off)/ relo )    )**2  )
                    else:
                        self.oo_res =    np.copy((   (np.diff(ec_model.off))    )**2   ) 
                        
            else:
                self.oo_res = 0            
            
            ec_res = np.append(ec_res, (self.vv_res)* w) # ec_model.wv)
            ec_res = np.append(ec_res, (self.oo_res)* w) # ec_model.wo)

        
        # T and EC joint simulation --> append residuals 
        if "T" in comp and "EC" in comp:
            res = np.append(ec_res, T_res  )
        elif comp == "T": 
            res = np.append(T_res, (self.vv_res)* w) # ec_model.wv)
        elif comp == "EC": 
            res = ec_res
        


         
        return res 
    
    def iter_fun(self, params, iteriter, res):        
        if iteriter != 0 and iteriter%20 == 0: 
            print(iteriter)