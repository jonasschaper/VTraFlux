# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:47:40 2024

@author: Jonas Schaper 
"""




import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern

def curvature_diff(ldf_single):
    outs = pd.DataFrame()  
    ldf_single = ldf_single.sort_values(by='w')
    ldf_single = ldf_single.reset_index(drop=True)

    xx = np.log10(ldf_single['w'])
        
    rho_hat =np.array(ldf_single['log10resnorm'])
    eta_hat =np.array(ldf_single['log10pnorm'])
    
    for i in range(1, len(xx) - 1):
        dx = xx[i+1] - xx[i]
        d2en = (eta_hat [i+1] - 2 * eta_hat[i] + eta_hat[i-1]) / (dx ** 2)
        d1en = (eta_hat[i+1] - eta_hat[i-1]) / ((1 * (xx[i+1] - xx[i-1])) ** 1)
        
        d2re = (rho_hat[i+1] - 2 * rho_hat[i] + rho_hat[i-1]) / (dx ** 2)
        d1re = (rho_hat[i+1] - rho_hat[i-1]) / ((1 * (xx[i+1] - xx[i-1])) ** 1)
        
        dff = pd.DataFrame({
            'xx': [xx[i]],
            'yres': [rho_hat[i]],
            'yen': [eta_hat[i]],
            'd1en': [d1en],
            'd2en': [d2en],
            'd1re': [d1re],
            'd2re': [d2re],
            'w': [ xx[i]]        })
        outs = pd.concat([outs, dff])
    
    outs['cur'] = (outs['d1re'] * outs['d2en'] - outs['d1en'] * outs['d2re']) / ((outs['d1en'] ** 2 + outs['d1re'] ** 2) ** (3/2))
    outs = outs.reset_index(drop=True)
     
    return outs

def load_run_info(directory, folder, postest):
    
    filename = "run_info.dat"
    if not os.path.exists(directory):
        print("Error: Directory '{}' does not exist.".format(directory))
        return None
    
    file_path = os.path.join(directory,folder, filename) # creating the full path to the text file
    
    
    if not os.path.isfile(file_path):
        print("Error: File '{}' does not exist in directory '{}'.".format(filename, directory))
        return None    
    try:
        with open(file_path, 'r') as file:

            runinfo = pd.read_csv(file_path,  delim_whitespace=True, header = None)  
            runinfo.columns =    ['param', 'value']
            log10w =  10**float(runinfo[runinfo['param'] == "weight:"]["value"])
            vres =  float(runinfo[runinfo['param'] == "vvstr:"]["value"])
            stt =  float(runinfo[runinfo['param'] == "ststr:"]["value"])

            header  = []; header.append("like")
            
            if runinfo[runinfo['param'] == "compsstr:"]["value"][0] == "EC":
                
                if  runinfo.at[3, 'value'] == "lt2": 
                    offset_vector = []
                    for i in range(1+1):
                        string = "ov_" + str(i)  
                        offset_vector.append(string)  
                
                if  runinfo.at[3, 'value'] == "v24": 
                    offset_vector = []
                    for i in range(int(float(runinfo.at[8, 'value'])+1)):
                        string = "ov_" + str(i) 
                        offset_vector.append(string) 
                
                if  runinfo.at[3, 'value'] == "c00": 
                
                    offset_vector = []
                    for i in range(1):
                        string = "ov_" + str(i)  
                        offset_vector.append(string)  

                num_vstr = int(  24/float(runinfo.at[2, 'value']) *float(runinfo.at[8, 'value']))#+1
                vv_vector = []
                for i in range(num_vstr+1):
                    string = "vv_" + str(i)  
                    vv_vector.append(string)  
                        
                header.extend(["al"])
                header.extend(vv_vector)
                header.extend(offset_vector)
                npars =1+len(vv_vector)+len(offset_vector)

                #header.extend(["rmse"  ,"sse" , "like_solo", "difo" , "difv" ,   "accepted" ])


            if runinfo[runinfo['param'] == "compsstr:"]["value"][0] == "EC.T":
                
                if  runinfo.at[3, 'value'] == "lt2": 
                    offset_vector = []
                    for i in range(1+1):
                        string = "ov_" + str(i)  
                        offset_vector.append(string)  
                
                if  runinfo.at[3, 'value'] == "v24": 
                    offset_vector = []
                    for i in range(int(float(runinfo.at[8, 'value'])+1)):
                        string = "ov_" + str(i) 
                        offset_vector.append(string) 
                
                if  runinfo.at[3, 'value'] == "c00": 
                
                    offset_vector = []
                    for i in range(1):
                        string = "ov_" + str(i)  
                        offset_vector.append(string)  

                num_vstr = int(  24/float(runinfo.at[2, 'value']) *float(runinfo.at[8, 'value']))#+1
                vv_vector = []
                for i in range(num_vstr+1):
                    string = "vv_" + str(i)  
                    vv_vector.append(string)  



                if len(runinfo) == 11:
                    n_thermal_pars = 4   
                    if runinfo.at[10, 'value'] == "TPO_01":
                        tpo_str = [  "pcs" , "kappas", "n", "beta"  ]
                    if runinfo.at[10, 'value'] == "TPO_02":
                        tpo_str = ["pcb" , "kappa_0", "n", "beta"  ]
                    if runinfo.at[10, 'value'] == "TPO_03":
                        tpo_str = [ "dds",  "yt"  ,"beta"  ]
                        n_thermal_pars = 3 
                   
                else:
                    tpo_str = [  "pcs" , "kappas", "n", "beta"  ]
                    n_thermal_pars = 4   

                header.extend(["al"])
                header.extend(vv_vector)

                header.extend(offset_vector)        
                header.extend(tpo_str)

                npars =1+n_thermal_pars+len(vv_vector)+len(offset_vector)
               
                    
            if runinfo[runinfo['param'] == "compsstr:"]["value"][0] == "T":
                
                print(file_path)
                # heat only run
                num_vstr = int(  24/float(runinfo.at[2, 'value']) *float(runinfo.at[8, 'value']))#+1
                vv_vector = []
                for i in range(num_vstr+1):
                    string = "vv_" + str(i)  
                    vv_vector.append(string)  
                header.extend(vv_vector)
                n_thermal_pars = 4                    
                
                if len(runinfo) == 11:
                    n_thermal_pars = 4   
                    if runinfo.at[10, 'value'] == "TPO_01":
                        tpo_str = [  "pcs" , "kappas", "n", "beta"  ]
                    if runinfo.at[10, 'value'] == "TPO_02":
                        tpo_str = ["pcb" , "kappa_0", "n", "beta"  ]
                    if runinfo.at[10, 'value'] == "TPO_03":
                        tpo_str = [ "dds",  "yt"  ,"beta"  ]
                        n_thermal_pars = 3 
                   
                else:
                    tpo_str = [  "pcs" , "kappas", "n", "beta"  ]
                    n_thermal_pars = 4  
                
                
                
                header.extend(tpo_str)
                
                #header.extend(["beta" , "pcs" , "kappas", "n"  ])
                npars =n_thermal_pars+len(vv_vector)
            
                
            current_header = copy.deepcopy(header)

            header.extend(["rmse"  ,"sse" , "like_solo", "difo" , "difv" ,   "accepted" ])
    
            if postest:
                # select if all DREAM runs have converged, i.e., reads in "postpars.dat"

                header = header
            else:
                # select if l-curve is simulated "on the fly", i.e., reads in "current_ext.dat"
                header = ['like', 'rmse', 'sse', 'like_solo', 'difo', 'difv']

        return header, log10w ,vres, npars, current_header, stt
    except Exception as e:
        print("Error reading file:", e)
        return None


def load_text_file(directory,folder, filename,postest):
    if not os.path.exists(directory):
        print("Error: Directory '{}' does not exist.".format(directory))
        return None
    
    file_path = os.path.join(directory,folder, filename)
    
    file_path2 = os.path.join(directory,folder, 'current.dat')

    
    if not os.path.isfile(file_path):
        print("Error: File '{}' does not exist in directory '{}'.".format(filename, directory))
        return None
    
    try:
        with open(file_path, 'r') as file:
            header, log10w ,vres, npars, current_header, stt = load_run_info(directory, folder, postest = postest)
            content = pd.read_csv(file_path,  delim_whitespace=True, header = None) 
            content.columns = header

            if os.path.isfile(file_path2):
                current = pd.read_csv(file_path2,  delim_whitespace=True, header = None)  
                
                if len(current.columns) > len(current_header):
                    current_header
                    current_header.extend(["kappa0"  ,"ddt" , "rrs" ])
                
                current.columns = current_header


            content['log10w'] = float(log10w)
            content['vres'] = vres
            content["difv"]
            
            content["pnorm"] = abs((content["like"]- content["like_solo"]))/(content["log10w"])


            test = 0
            if test== 0:
                
                
                if postest == True:
                    param = copy.deepcopy(content)
                elif postest == False and  os.path.isfile(file_path2):
                    param = copy.deepcopy(current)

                int(24/vres)
                vvs = param[[col for col in param.columns if "vv" in col]]
                
                vvs_24 = vvs.iloc[:, int(stt/vres+1):]
                diffs = vvs_24.diff(axis=1) # exludes 0
                realmeans = abs(vvs_24.mean(axis = 1))

                vsum = np.sum(np.array((diffs.fillna(0).div(realmeans, axis=0))**2), axis=1)

                oos = param[[col for col in param.columns if "ov" in col]]
                diffs = oos.diff(axis=1)
                realmeans = abs(oos.mean(axis = 1))

                osum = np.sum(np.array((diffs.fillna(0).div(realmeans, axis=0))**2), axis=1)


                sumsum = vsum + osum
                content["pnorm"]  = sumsum
                
                
            content["resnorm"] = (content["like_solo"]*-2)
            
            # Calculate column means and standard deviations
            means = np.log10(content["resnorm"].mean())
            means_resnorm = np.log10(content["resnorm"]).mean()

            ww = (content["log10w"]).mean()

            std_resnorm = np.log10(content["resnorm"]).std()
                        
            means = np.log10(content["pnorm"]).mean()

            std_devs = np.log10(content["pnorm"]).std()

            content["AIC"] = 2*npars- 2* content["like"]
            AIC = content["AIC"].mean()
            AIC_sd = content["AIC"].std()

            errors = content["resnorm"].std()/(np.log(10)* content["resnorm"].mean())

            # Create a new DataFrame to store the results
            result_df = pd.DataFrame({'log10resnorm': np.log10(content["resnorm"]).mean(), 
                                      'log10resnorm_std' :errors,
                                      'log10pnorm': np.log10(content["pnorm"]).mean(),
                                      'log10pnorm_std' :np.log10(content["pnorm"]).std(), 
                                     
                                      'AIC' :AIC,
                                      'AIC_std' :AIC_sd,
                                      'like' :(content["like"]).mean(),
                                      'like_sd' :(content["like"]).std(),
                                      'rmse' :(content["rmse"]).mean(),
                                      'rmse_sd' :(content["rmse"]).std(),
                                      'likesolo': (content["like_solo"]).mean(),
                                      'likesolo_sd': (content["like_solo"]).std(),
                                      'resnorm' :(content["resnorm"]).mean(),
                                      'resnorm_std' :(content["resnorm"]).std(),
                                      'pnorm' :(content["pnorm"]).mean(),
                                      'pnorm_std' :(content["pnorm"]).std(),
                                      "w": ww, "vres": vres}, index=[0])
        
            

            
        return content, result_df
    except Exception as e:
        print("Error reading file:", e)
        return None






def process_folders(directory, filename, postest):
    if not os.path.exists(directory):
        print("Error: Directory '{}' does not exist.".format(directory))
        return
    
    items = os.listdir(directory)
    folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    
    result_df = pd.DataFrame(columns=['resnorm', 'resnorm_std',"pnorm","pnorm_std","vres","w"])    # Loop through the folder names
    for folder in folders:
        print("Processing folder:", folder)
        
        postpars , ldf= load_text_file(directory,folder, filename = filename, postest = postest)
        result_df = pd.concat([result_df, ldf], ignore_index=True)
            
    result_df = result_df.sort_values(by='w', ascending=True)
    result_df = result_df.reset_index(drop=True)
    
        
    return postpars, folder, result_df

def plt_Lcurve(subset, dirName, erfac, showplt ):
    pd.set_option('display.max_columns', 500)
    
    
    subset = subset.sort_values(by='w', ascending=True)
    subset = subset.reset_index(drop=True)
    
    
    
    X_train = subset.log10pnorm
    y_train =  subset.log10resnorm

    
    errs  =  np.array( subset.log10resnorm_std)
    if np.isnan(subset.log10resnorm_std).all():
        errs = np.array(subset.log10resnorm_std.fillna(0))

      
    x_data = np.linspace(min(X_train), max(X_train), 1000)#.reshape(-1, 1)
    
    # increae error in matern so that the model can also avoid going through points 
    kernel = 1 * Matern(length_scale=.05, length_scale_bounds=(1e-05, 100000.0), nu=1.5)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True, alpha=(erfac*errs)**2)
    gaussian_process.fit((np.atleast_2d(X_train).T), (y_train))
    mean_prediction, std_prediction = gaussian_process.predict((np.atleast_2d(x_data).T), return_std=True)
    
    labels = np.log10((subset.w))
    plt.rcParams['figure.dpi']=600
    plt.plot(X_train, y_train, label=r"Linear Interpolation", linestyle="dotted")
    plt.scatter(X_train, y_train, label="L-curve point estimates")
    plt.errorbar(X_train, y_train, yerr=subset.log10resnorm_std, fmt="o")
    plt.errorbar(X_train, y_train,xerr=subset.log10pnorm_std, fmt="o")
    for i, label in enumerate(labels):
        plt.annotate(str(round(label,3)), (np.array(X_train)[i], np.array(y_train)[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.plot(x_data, mean_prediction, label="Mean prediction GPR")
    plt.fill_between(
        x_data.ravel(),    mean_prediction - 1.96 * std_prediction, mean_prediction + 1.96 * std_prediction, alpha=0.5, label=r"95% confidence interval",)
    plt.legend()
    plt.ylabel(r'$\log_{10}( \| e_{rel} \| _{2}^{2} )$')
    plt.xlabel(r'$\log_{10}( \| \Delta x_{rel} \| _{2}^{2} )$')
    _ = plt.title("L-curve - gaussian process regression")
    plt.savefig(   (dirName + "/" + 'lcurve' + '.png'))
    plt.close()

    d = {'x': x_data, 'pred': mean_prediction, 'pstd': std_prediction}
    df = pd.DataFrame(data=d)
    if showplt:
        os.startfile(dirName + "/" + 'lcurve' + '.png')
    
    return df


def estimate_w(result_df, smooth_l, dirName, mco):
    

    
    result_df = result_df.sort_values(by='w', ascending=True)
    result_df = result_df.reset_index(drop=True)
    
    
    cur_file = os.path.join(dirName, "smooth_lcurve.txt")
    smooth_l.to_csv(cur_file, sep='\t', index=False, encoding='utf-8', float_format='%.5f')
    
    
    
    
    outs = pd.DataFrame()  # Initialize an empty DataFrame
    for entry in range(len(result_df["w"])):
        
            current_x = result_df.loc[entry,"log10pnorm"]
            
            difference_array = np.absolute(smooth_l.x - current_x)
     
            # find the index of minimum element from the array
            index = difference_array.argmin()
            
            
            dff = pd.DataFrame({
                'log10pnorm': [result_df.loc[entry,"log10pnorm"]],
                'AIC': [result_df.loc[entry,"AIC"]],
                'AIC_sd': [result_df.loc[entry,"AIC_std"]],
                'like': [result_df.loc[entry,"like"]],
                'like_sd': [result_df.loc[entry,"like_sd"]],
                'rmse': [result_df.loc[entry,"rmse"]],
                'rmse_sd': [result_df.loc[entry,"rmse_sd"]],

                'log10resnorm': [smooth_l.loc[index,"pred"]  ],
                #'w': [ np.log10(result_df.loc[entry,"w"])]        })
                'w': [ result_df.loc[entry,"w"]]        })
            outs = pd.concat([outs, dff])
    
    outs = outs.fillna(0)

        
    #plt.plot(outs.log10resnorm, outs.w, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    #plt.plot(outs.log10resnorm, outs.log10pnorm, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    outs = outs.reset_index(drop=True)
    cur_w = curvature_diff(outs)
    outs["ww"]= np.log10(outs.w)
        

    if mco == "MCO_01":
        # estimated curvature central difference of L-curve points 
        raw_curve = curvature_diff(result_df)

    
        index = raw_curve.cur.argmax()
        ideal_w =      10**raw_curve.loc[index,"w"]     
    
        ideal_AIC =  outs.loc[ round((outs["ww"]),4) == float(round(raw_curve.loc[index,"w"],1)) ,   ]    



    if mco == "MCO_02":

        # estimated curvature based on GRP approximation of L-cuve 
        index = cur_w.cur.argmax()
        ideal_w =      10**cur_w.loc[index,"w"]     
        ideal_AIC =  outs.loc[ round((outs["w"]),4) == round(float(10**cur_w.loc[index,"w"]),4) ,   ]    
    
        cur_file = os.path.join(dirName, "est_lcurve.txt")
        outs.to_csv(cur_file, sep='\t', index=False, encoding='utf-8', float_format='%.5f')
        
        cur_file = os.path.join(dirName, "curve_log10w_type2.txt")
        cur_w.to_csv(cur_file, sep='\t', index=False, encoding='utf-8', float_format='%.5f')
         

    aic_str = str(round(np.array(ideal_AIC["AIC"])[0],)) +  "_" +  str(round(np.array(ideal_AIC["AIC_sd"])[0],))
    like_str = str(round(np.array(ideal_AIC["like"])[0],)) +  "_" +  str(round(np.array(ideal_AIC["like_sd"])[0],))
    rmse_str = str(round(np.array(ideal_AIC["rmse"])[0]*1000,1)) +  "_" +  str(round(np.array(ideal_AIC["rmse_sd"])[0]*1000,1))

    
    f = open(dirName +  '/'+'ideal_weight'  +'.dat','w')
    f.write('%s ' % "w")
    f.write('%g ' % ideal_w)    
    f.write('\n')
    f.write('%s ' % "AIC")
    f.write('%s ' % aic_str)
    f.write('\n')
    f.write('%s ' % "rmse")
    f.write('%s ' % rmse_str)
    f.write('\n')
    f.write('%s ' % "total like")
    f.write('%s ' % like_str)
    f.write('\n')
    f.close()

    
    showplt = False
    if showplt:
        os.startfile(dirName +  '/'+'ideal_weight'  +'.dat')
    

    return cur_w





def find_weight(directory, erfac =10,  postest = True, mco = "MCO_02", printres = True):
    
    
    if postest: 
        filename =  "postpars.dat"
    else:
        filename =  "current_ext.dat"

    postpars, folder ,result_df= process_folders(directory = directory, filename = filename, postest = postest) # select postest == True if dream run has finished; False if on-the-fly l curve simulation
    smooth_l  = plt_Lcurve(subset =result_df,dirName = directory, erfac = erfac, showplt = printres) # estimates a smooth L-curve 
    cur_w =estimate_w(result_df, smooth_l, dirName = directory, mco = mco) # using curvature with respect to w 































