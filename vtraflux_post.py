# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:59:06 2024

@author: Jonas
"""

from regular_vtraflux import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_thermal(postpars): 
    
    
    df = postpars

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # Plot violin plots on the subplots
    sns.violinplot(data=df[['pcs']], ax=axes[0, 0])
    axes[0, 0].set_title('Group 1')

    sns.violinplot(data=df[['kappas']], ax=axes[0, 1])
    axes[0, 1].set_title('Group 2')

    sns.violinplot(data=df[['n']], ax=axes[1, 0])
    axes[1, 0].set_title('Group 3')

    sns.violinplot(data=df[['beta']], ax=axes[1, 1])
    axes[1, 1].set_title('Group 4')

    plt.tight_layout()
    plt.show()



def plot_ts(directory, folder, filename):

    
    file_path = os.path.join(directory,folder, 'T_cmod_all.txt')
    file_path = os.path.join(directory,folder, filename)
    
    #file_path2 = os.path.join(directory,folder, 'T_cmod_all.dat')

    
    if not os.path.isfile(file_path):
        print("Error: File '{}' does not exist in directory '{}'.".format(filename, directory))
        return None
    try:
        with open(file_path, 'r') as file:

            content = pd.read_csv(file_path,  delim_whitespace=True, header = None)  # You may need to adjust the arguments based on the file format
            
            pltdf = pd.DataFrame()
    
            pltdf["modt"] = content[0]
            del content[content. columns[0]]
            
            reali =  content
            pltdf["row_mea"] = reali.mean(axis=1)
            pltdf["row_sdd"] = reali.std(axis=1)
            
            plt.plot(pltdf["modt"],  pltdf["row_mea"], label='mean')
            plt.fill_between(pltdf["modt"],  pltdf["row_mea"] -    pltdf["row_sdd"],  pltdf["row_mea"] +    pltdf["row_sdd"], color='blue', alpha=0.2)
            plt.xlabel('model time (h)')
            plt.ylabel('Temperature °C')
            plt.legend()
            plt.show()
            
            
    except Exception as e:
        print("Error reading file:", e)
        return None     
    
    return pltdf


secondraft = "C:/Users/Jonas/Documents/Projects/tempts/modeling/transnum/finalfinal" 
resdir = "C:/Users/Jonas/Dropbox/vtraflux_v01_final/dreamres" 

resdir = "C:/Users/Jonas/Dropbox/vtraflux_v01_upload_muscheln/modres" 

#directory =  os.path.join(secondraft,"ECT", "EC.T_inS6_01_1_8_vv06st00_ov24_2030_prim")
#directory =  os.path.join(secondraft,"EC_inS6_01_1_6_vv24st00_oc00_19_prib")

directory =  os.path.join(resdir,"T_inS6_01_1_5_vv06st00_2030_priw")

runrun = "T_inS6_01_1_5_vv06st00_2030_priw_w-2.0"
postpars , ldf= load_text_file(directory, folder = 'T_inS6_01_1_5_vv06st00_2030_priw_w-2.0', filename = "postpars.dat", postest = True)
postpars , ldf= load_text_file(directory, folder = runrun, filename = "postpars.dat", postest = True)

# could also import them from process_folders
# postpars, folder ,result_df= process_folders(directory = directory, filename = "postpars.dat", postest = True) # select postest == True if dream run has finished; False if on-the-fly l curve simulation


plot_ts(directory, folder =  runrun, filename =  'T_cmod_all.txt')
plot_ts(directory, folder =  runrun, filename =  'T_v_all.txt')
plot_ts(directory, folder =  runrun, filename =  'T_d_all.txt')




plot_thermal(postpars)



