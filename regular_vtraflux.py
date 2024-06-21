# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:47:40 2024

@author: Jonas Schaper 
"""




import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(ddx * dy - dx * ddy) / ((dx ** 2 + dy ** 2) ** 1.5)
    curvature = (ddx * dy - dx * ddy) / ((dx ** 2 + dy ** 2) ** 1.5)
    return curvature

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

            if postest:
                # select if all DREAM runs have converged, i.e., reads in "postpars.dat"
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


                else: 
                    # select if l-curve is simulated "on the fly", i.e., reads in "current_ext.dat"
                    num_vstr = int(  24/float(runinfo.at[8, 'value']) *float(runinfo.at[2, 'value']))
                    vv_vector = []
                    for i in range(num_vstr+1):
                        string = "vv_" + str(i)  
                        vv_vector.append(string)  
                    header.extend(vv_vector)
                    header.extend(["beta" , "pcs" , "kappas", "n"  ])
                header.extend(["rmse"  ,"sse" , "like_solo", "difo" , "difv" ,   "accepted" ])
                
            else:
                header = ['like', 'rmse', 'sse', 'like_solo', 'difo', 'difv']
        
        return header, log10w ,vres
    except Exception as e:
        print("Error reading file:", e)
        return None


def load_text_file(directory,folder, filename,postest):
    if not os.path.exists(directory):
        print("Error: Directory '{}' does not exist.".format(directory))
        return None
    
    file_path = os.path.join(directory,folder, filename)
    
    if not os.path.isfile(file_path):
        print("Error: File '{}' does not exist in directory '{}'.".format(filename, directory))
        return None
    
    try:
        with open(file_path, 'r') as file:
            header, log10w ,vres= load_run_info(directory, folder, postest = postest)
            print(header)

            content = pd.read_csv(file_path,  delim_whitespace=True, header = None)  # You may need to adjust the arguments based on the file format
            print(len(content.columns))
            print(file_path)
            print(content)


            content.columns = header
            content['log10w'] = float(log10w)
            content['vres'] = vres
            content["difv"]
            
            content["pnorm"] = abs((content["like"]- content["like_solo"]))/(content["log10w"])

            content.loc[content['pnorm']  == 0, "pnorm"] = 0.00001
            content["resnorm"] = (content["like_solo"]*-2)
            
            # Calculate column means and standard deviations
            means = np.log10(content["resnorm"].mean())
            means_resnorm = np.log10(content["resnorm"]).mean()

            ww = (content["log10w"]).mean()

            std_resnorm = np.log10(content["resnorm"]).std()
                        
            means = np.log10(content["pnorm"]).mean()
            std_devs = np.log10(content["pnorm"]).std()


            errors = content["resnorm"].std()/(np.log(10)* content["resnorm"].mean())

            # Create a new DataFrame to store the results
            result_df = pd.DataFrame({'log10resnorm': np.log10(content["resnorm"]).mean(), 
                                      'log10resnorm_std' :errors,
                                      'log10pnorm': np.log10(content["pnorm"]).mean(),
                                      'log10pnorm_std' :np.log10(content["pnorm"]).std(), 
                                      
                                      'resnorm' :(content["resnorm"]).mean(),
                                      'resnorm_std' :(content["resnorm"]).std(),
                                      'pnorm' :(content["pnorm"]).mean(),
                                      'pnorm_std' :(content["pnorm"]).std(),
                                      "w": ww, "vres": vres}, index=[0])
            
        return content, result_df
    except Exception as e:
        print("Error reading file:", e)
        return None



def get_header_old(folder):
    parts = folder.split("_")
    log10w = parts[-1][-len(parts[-1])+1:] #parts[-1][-3:] 
    header  = []; header.append("like")

    if parts[0][0:2] == "EC":
        
        if parts[-4][1] == "v": 
            num_strings = int(24/float(parts[-4][-2:])*float(parts[-6]))#+1
            offset_vector = []
            for i in range(num_strings+1):
                string = "ov_" + str(i) 
                offset_vector.append(string)  
        
        num_vstr = int(  24/float(parts[-5][2:4]) *float(parts[-6]))#+1
        vv_vector = []
        for i in range(num_vstr+1):
            string = "vv_" + str(i)  
            vv_vector.append(string)  
                
        header.extend(["al"])
        header.extend(vv_vector)
        header.extend(offset_vector)
        if parts[0][3] == "T":
            header.extend(["beta" , "pcs" , "kappas", "n"  ])
        vres = float(parts[-5][2:4])

    else: 
        num_vstr = int(  24/float(parts[-4][2:4]) *float(parts[-5]))#+1
        vv_vector = []
        for i in range(num_vstr+1):
            string = "vv_" + str(i)  
            vv_vector.append(string) 
                
        header.extend(vv_vector)
        header.extend(["beta" , "pcs" , "kappas", "n"  ])
        vres = float(parts[-4][2:4])

    header.extend(["rmse"  ,"sse" , "like_solo", "difo" , "difv" ,   "accepted" ])
    
    return header  , log10w, vres
          
def get_header(folder):

    parts = folder.split("_")
    log10w = parts[-1][-len(parts[-1])+1:] #parts[-1][-3:] 
    header  = []; header.append("like")
    

    if parts[0][0:2] == "EC":
        
        if parts[-4][1] == "v": 
            num_strings = int(24/float(parts[-4][-2:])*float(parts[-6]))#+1
            offset_vector = []
            for i in range(num_strings+1):
                string = "ov_" + str(i)  
                offset_vector.append(string)  
        
        num_vstr = int(  24/float(parts[-5][2:4]) *float(parts[-6]))#+1
        vv_vector = []
        for i in range(num_vstr+1):
            string = "vv_" + str(i)  
            vv_vector.append(string)  
                
        header.extend(["al"])
        header.extend(vv_vector)
        header.extend(offset_vector)
        vres = float(parts[-5][2:4])

    else: 
        num_vstr = int(  24/float(parts[-4][2:4]) *float(parts[-5]))#+1
        vv_vector = []
        for i in range(num_vstr+1):
            string = "vv_" + str(i) 
            vv_vector.append(string) 
        header.extend(vv_vector)
        header.extend(["beta" , "pcs" , "kappas", "n"  ])
        vres = float(parts[-4][2:4])

    
    header  = []; header.append("like")
    header.extend(["rmse"  ,"sse" , "like_solo", "difo" , "difv" ])
    
    return header  , log10w, vres


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

    return postpars, folder, result_df

def plt_Lcurve(subset, dirName, erfac, showplt ):
    pd.set_option('display.max_columns', 500)
    
    X_train = subset.log10resnorm
    y_train =  subset.log10pnorm
    
    X_train = subset.log10pnorm
    y_train =  subset.log10resnorm
    
    
    errs  =  np.array( subset.log10resnorm_std)
    
      
    x_data = np.linspace(min(X_train), max(X_train), 1000)#.reshape(-1, 1)
    
    # increae error in matern so that the model can also avoid going through points 
    kernel = 1 * Matern(length_scale=.05, length_scale_bounds=(1e-05, 100000.0), nu=1.5)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True, alpha=(erfac*errs)**2)
    gaussian_process.fit((np.atleast_2d(X_train).T), (y_train))
    mean_prediction, std_prediction = gaussian_process.predict((np.atleast_2d(x_data).T), return_std=True)
    
    labels = (subset.w)
    plt.plot(X_train, y_train, label=r"linear interpilation", linestyle="dotted")
    plt.scatter(X_train, y_train, label="Observations")
    for i, label in enumerate(labels):
        plt.annotate(str(round(label,1)), (np.array(X_train)[i], np.array(y_train)[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.plot(x_data, mean_prediction, label="Mean prediction")
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

def estimate_w(result_df, smooth_l, dirName):
    
    
    
    result_df = result_df.sort_values(by='w', ascending=True)
    result_df = result_df.reset_index(drop=True)
    
    raw_curve = curvature_diff(result_df)
    
    outs = pd.DataFrame()  # Initialize an empty DataFrame
    for entry in range(len(result_df["w"])):
        
            current_x = result_df.loc[entry,"log10pnorm"]
            difference_array = np.absolute(smooth_l.x - current_x)
            index = difference_array.argmin()             


            dff = pd.DataFrame({
                'log10pnorm': [result_df.loc[entry,"log10pnorm"]],
                'log10resnorm': [smooth_l.loc[index,"pred"]  ],
                #'w': [ np.log10(result_df.loc[entry,"w"])]        })
                'w': [ result_df.loc[entry,"w"]]        })
            outs = pd.concat([outs, dff])
    
        

    outs = outs.reset_index(drop=True)
    cur_w = curvature_diff(outs)

    cur_file = os.path.join(dirName, "curve_log10w_type2.txt")
    cur_w.to_csv(cur_file, sep='\t', index=False, encoding='utf-8', float_format='%.5f')
     
    cur_file = os.path.join(dirName, "curve_log10w_type1.txt")
    raw_curve.to_csv(cur_file, sep='\t', index=False, encoding='utf-8', float_format='%.5f')
    
    cur_file = os.path.join(dirName, "smooth_lcurve.txt")
    smooth_l.to_csv(cur_file, sep='\t', index=False, encoding='utf-8', float_format='%.5f')
    
    cur_file = os.path.join(dirName, "raw_lcurve.txt")
    result_df.to_csv(cur_file, sep='\t', index=False, encoding='utf-8', float_format='%.5f')
    
    
    
    #curvature( outs.log10resnorm, outs.w)[1:-1]
    index = cur_w.cur.argmax()
    ideal_w =      10**cur_w.loc[index,"w"]     

    
    f = open(dirName +  '/'+'ideal_weight'  +'.dat','w')
    f.write('%g ' % ideal_w)

    f.write('\n')
    f.close()

    
    return cur_w


def find_weight(smooth_l,result_df, dirName):
    # calculate curvature from modelled L-curve
    smooth_l['cur'] = curvature( smooth_l.pred, smooth_l.x)


    max_cur = smooth_l.cur.idxmax()
    maxcur_df = smooth_l.iloc[max_cur,]

    best_x = maxcur_df.x

    # locate index of best x value and correspoinding weight / which of estimated cases is closest to best x 
    difference_array = np.absolute(result_df.log10pnorm - best_x)
     # find the index of minimum element from the array
    index = difference_array.argmin()
    ideal_w =      result_df.loc[index,"w"]     
    

    f = open(dirName +  '/'+'ideal_weight'  +'.dat','w')
    f.write('%g ' % ideal_w)

    f.write('\n')
    f.close()

