# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:33:12 2024

@author: Jonas
"""

import os
import pandas as pd
from datetime import datetime
import copy
import numpy as np


def write_conc_ts_txt(comps, indat, str_time, end_time, output_dir, depths):
    os.makedirs(output_dir, exist_ok=True)
    print(indat['datetime'].iloc[0].tzinfo)
    
    for selcomp in comps:
        print(selcomp)
        
        st_f = pd.to_datetime(str_time).tz_localize(indat['datetime'].iloc[0].tzinfo)
        et_f = pd.to_datetime(end_time).tz_localize(indat['datetime'].iloc[0].tzinfo)
        
        
        #        df.Date = et_f.Date.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        
        for depth in depths:
            input01 = indat[(indat['variable'] == selcomp) & (indat['depth_cm'] == depth)]
            test = input01['datetime']
            
            start_it = (test - st_f).abs().idxmin()
            end_it = (test - et_f).abs().idxmin()
            
            mins = abs(round((input01['datetime'].iloc[0] - input01['datetime'].iloc[1]).total_seconds() / 60))
            input01s = input01.loc[start_it:end_it]
            
            #input01s =copy.deepcopy(input01.loc[start_it:end_it])
        
            
            comp_03 = write_conc(selcomp, f"{depth:02d}", input01s, ['timestep', 'value'], mins, output_dir)
            
def write_conc(printcomp, printdepth, input_data, col_names, mins, output_dir):

    input_data = input_data.reset_index(drop=True)

    nrows = len(input_data.index)


    # Assuming 'input' is a pandas DataFrame and 'mins' is a defined variable
    #mins = 10  # Example value, replace with the actual value you are using
    
    # Calculate hours_num
    hours_num = np.round(np.arange(0, nrows * mins, mins) / 60, 4)
    
    # Calculate the length of hours_num
    len_hours_num = len(hours_num)
    
    # Filter hours_num to include only non-negative values and format to 4 decimal places
    hours = ["{:.4f}".format(hour) for hour in hours_num if hour >= 0]
    
    hours[-1]
    # Calculate the length of hours
    len_hours = len(hours)

    input_data['timestep'] = hours

    timesteps = len(input_data['timestep'])
    
    #"{:.4f}".format(timesteps)
    
    if printcomp == "Temp_C": statevar = "T"
    if printcomp == "EC_mScm": statevar = "EC"

    file_name = f"{statevar}_{printdepth}.txt"
    file_path = os.path.join(output_dir, file_name)
    col_names = "{:.0f}".format(timesteps) 
    
    input_data2 = input_data.loc[:, ['timestep', "value"]] 
    dim = np.shape(input_data2)

    input_data.loc[0,'timestep']

    with open(file_path, 'w') as f:
        # Write the header manually
        f.write('%g ' % timesteps)
        f.write('\n')
        for k in range(dim[0]):
            f.write('%s         %s \n' %  (input_data.loc[k,'timestep'], str("{:.4f}".format(input_data.loc[k,'value']))) )        
        #f.write('\n')
        f.close()
        
        #input_data2.to_csv(f, columns=None, index=False, sep='\t')

    return input_data


outdir_inS6 = "C:/Users/jonas/Documents/Projects/tempts/modeling/transnum/indata/inS6_01/"

# Loading the data
filefile = "C:/Users/Jonas/Dropbox/vtraflux_v01_final/rawdata/temp_EC_timeseries.csv"
vtraflux_data = pd.read_csv(filefile, sep=";", header=0)

# subset erpe2016 dataset
erpe2016 = vtraflux_data[vtraflux_data['dataset'] == "erpe2016"]
erpe2016 = erpe2016.reset_index(drop=True)



# subset temp data 
erpe2016_T = erpe2016[erpe2016['variable'] == "Temp_C"]
erpe2016_T['depth_cm'] = erpe2016_T['depth_cm'].astype(float)
erpe2016_T['datetime'] = pd.to_datetime(erpe2016_T['datetime'], format="%Y-%m-%d %H:%M:%S").dt.tz_localize(None)

write_conc_ts_txt(comps=["Temp_C"], indat=erpe2016_T, str_time="2016-04-23 00:00:00 CEST", end_time="2016-04-30 23:50:00 CEST", output_dir=outdir_inS6, depths=[0, 5, 10, 15, 20, 30, 50])


erpe2016_ec = erpe2016[erpe2016['variable'] == "EC_mScm"]
erpe2016_ec = erpe2016_ec.reset_index(drop=True)
erpe2016_ec['depth_cm'] = pd.to_numeric(erpe2016_ec['depth_cm'])
erpe2016_ec['datetime'] = pd.to_datetime(erpe2016_ec['datetime'], format="%Y-%m-%d %H:%M:%S").dt.tz_localize(None)
write_conc_ts_txt(comps=["EC_mScm"], indat=erpe2016_ec, str_time="2016-04-23 00:00:00 CET", end_time="2016-04-30 23:55:00 CET", output_dir=outdir_inS6, depths=[0, 19])

#pd.to_datetime(erpe2016_T['datetime'], format="%Y-%m-%d %H:%M:%S").dt.tz_localize('Europe/Berlin')

#erpe2016_T['depth_cm'] = np.copy(pd.to_numeric(erpe2016_T['depth_cm']))
#erpe2016_T
#erpe2016_T['depth_cm'] = 

#erpe2016_T['Date'] = erpe2016_T['datetime']



pd.to_datetime(erpe2016_ec['datetime'], format="%Y-%m-%d %H:%M:%S").dt.tz_localize('Europe/Berlin')

#erpe2016_ec['Date'] = pd.to_datetime(erpe2016_ec['datetime'], format="%Y-%m-%d %H:%M:%S").dt.tz_localize('Europe/Berlin')


