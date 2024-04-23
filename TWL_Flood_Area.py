# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# ** This script intend to generate time-series files for TWL flood area
# ** Use "Dist_Flood_hol86" for every points in every area, then approximately calculate the areas by considering area=200*max_dist
# ** Edit by Yi Hong, 08/15/2023
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    
import os
# import xarray as xr
import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
# from netCDF4 import num2date
# from math import cos, asin, sqrt
import datetime
import matplotlib.dates as md


# import nctoolkit    # https://nctoolkit.readthedocs.io/en/latest/
#%%
# ==================================================================================
# ===============================USER CONTROLS======================================
# ==================================================================================  
 
# option for saving directory
path_Nodes = r'C:\FVCOM_Mich\Post_Scripts\Nodes_WL'
# bbox = os.path.join(path_Nodes, 'bbox_limits.csv')
# data_bbox = pd.read_csv(bbox, sep=',', index_col=0)

path_out = r'C:\FVCOM_Mich\Post_Scripts\Flood_comparisons'

path_predata=r'C:\Research_Mich\TWL_Flood\Py_script\Precalculated_data' 
path_vec_elev = os.path.join(path_predata, 'Vectors_Elev_Combined')
dir_TWL = r'U:\Research_Mich\TWL_Flood\Outputs\20200426'

Outfig=r'C:\FVCOM_Mich\FVCOM_runs\Fig_extend\Fig_TimeSerie\TWL'

# nodes_coast = pd.read_csv(os.path.join(path_Nodes, 'Nodes_coast.csv'), sep=',')

#%% Flood zone nodes
nodes_lud = [23239, 22998, 22997, 22996, 22760, 22759, 22758, 22757, 22756, 22755, 22753, 22497, 22491, 22222, 21944, 21945, 21946, 21656, 21655]
nodes_mont = [16265, 16264, 16263, 16262, 16261, 16028, 16027, 16026, 16025, 16024, 16023, 16022, 16020]
nodes_muskeg = [14701, 14700, 14699, 14698, 14697, 14696, 14695, 14420, 14421, 14422, 14423, 14426, 14427, 14425, 14424, 14155, 14143, 13867, 13866, 13865, 13864, 13863, 13862]
nodes_grandH = [13323, 13110, 13109, 13108, 12838, 12834, 12833, 12561]
nodes_holland = [9936, 9935, 9934, 9933, 9932, 9931, 9927, 9926, 9638, 9637, 9636, 9634, 9633, 9338, 9337, 9023, 9022]
nodes_southH = [7011, 6716, 6715, 6714, 6711, 6712, 6713, 6710, 6709, 6418, 6417, 6416, 6415, 6414, 6412, 6096, 6095]
nodes_stjose = [4086, 4094, 4093, 4092, 4091, 4090, 4089, 4088, 4087, 4085, 4084, 4083, 4082, 4081, 4080, 4079, 4078, 4077, 4076, 4075, 3743, 3744, 3745, 3748, 3749]
nodes_whChi = [2140, 2139, 2078, 2031, 2030, 1984, 1985, 2033, 2079, 2148, 2224, 2335, 2451, 2570, 2691, 2692, 2693, 2696, 2697, 2695, 2571, 2454, 2337, 2336, 2452, 2338, 2225, 2149, 2081, 2080, 2082, 2091, 2090, 2089, 2088, 2087, 2086, 2085, 2084, 2083, 2093, 2092, 2161, 2165, 2166, 2167, 2168, 2169]
nodes_whChi.extend(list(range(2252, 2241, -1)))
nodes_greenB = [1, 2, 3, 4, 9, 10, 13, 14, 17, 18, 21, 22, 26, 27, 28, 29, 30, 34, 35, 38, 39, 40, 43, 44,45, 46, 52, 53, 56, 57, 58, 62, 66, 69, 70, 72, 98, 99, 100 ,101, 103, 104, 105, 106, 188, 187, 224, 226, 259, 183, 148, 115]
nodes_greenB.extend(list(range(318, 304, -1)))
nodes_greenB.extend(list(range(80, 97)))

data_nodes = {'GreenBay': nodes_greenB, 'Ludington': nodes_lud, 'Muskegon': nodes_muskeg,'Montague':nodes_mont, 
              'Holland':nodes_holland, 'Whiting':nodes_whChi}

#%%
# =============================================================================
# TWL outputs
# =============================================================================
lst_ts =[]
start_time = datetime.datetime(2020, 4, 26, 1, 0)  # Change this to your desired start time
end_time = datetime.datetime(2020, 5, 22, 0, 0)   # Change this to your desired end time
time_step = datetime.timedelta(hours=1)
current_time = start_time

while current_time <= end_time:
    lst_ts.append(current_time)
    current_time += time_step


for zone, nodes in data_nodes.items():
    # zone, nodes =  next(iter(data_nodes.items()))
# Creat dataframe
    df_flood_area = pd.DataFrame(index=lst_ts,columns=nodes)
#%
    for j in nodes:
        #%     
        TWL_nc = Dataset(os.path.join(dir_TWL, str(j)+'.nc'), 'r')
        Dist_Flood_hol86 = TWL_nc.variables['Dist_Flood_hol86'][:]  
        min_dist = np.nanmin(Dist_Flood_hol86)
        Dist_Flood_hol86 = Dist_Flood_hol86 - min_dist
        df_flood_area[j] = Dist_Flood_hol86
    
    df_flood_area['Sum_Area'] = df_flood_area.sum(axis=1)  
    last_half_columns = df_flood_area.iloc[:, int(len(df_flood_area.columns) * 0.5):len(df_flood_area.columns)-1]
    df_flood_area['Last_50_Sum'] = last_half_columns.sum(axis=1)
    
    df_flood_area.to_csv(os.path.join(path_Nodes,'TWL\\TWL_FloodTS_'+zone+'.csv'), sep=',', index=True)  
    #%
    fig1=plt.figure(figsize=(10.0,6.0))
    ax1=plt.gca()

    plt.plot(df_flood_area.index, df_flood_area['Sum_Area']*0.02, label='Flood Area',linestyle='-',color='b',lw=2)   # to ha
    plt.plot(df_flood_area.index, df_flood_area['Last_50_Sum']*0.02, label='Selected Flood Area',linestyle='-',color='k',lw=2)   # to ha 
    ax1.set_ylabel('Flooding area (ha)',fontsize=14, weight='bold')
    ax1.set_xlabel('Date',fontsize=14, weight='bold')
    ax1.legend(loc='best',fontsize=14, ncol=1).set_title('')

    plt.title('Flooding area for '+zone,fontsize=16, weight='bold')
    ax1.tick_params(axis='both', labelsize=14)

    xfmt = md.DateFormatter('%m-%d')
    ax1.xaxis.set_major_formatter(xfmt)

    fig_name1=os.path.join(Outfig,'Flood_TWL_'+zone+'.jpg')
    plt.savefig(fig_name1, dpi=300)
#%% test plot of flood area        

  