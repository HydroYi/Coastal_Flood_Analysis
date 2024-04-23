# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# ** Compare the flood simulations with LMHOFS and extended mesh grids
# ** R scripts can also be used to plot maps
# ** A bash script is available for downloading data
# ** Edit by Yi Hong, 05/12/2021, 
# ** Modified by Yi Hong, 08/16/2023, add TWL and made modifications
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
'''
Plot maps with PyFVCOM package

    '''
    
import os
# import xarray as xr
import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from math import cos, asin, sqrt
import datetime
import matplotlib.dates as md

# os.environ["PROJ_LIB"] = r"C:\Users\yhon\Anaconda3\Library\share"; #path of the proj for basemap, not useful on HPC
# os.environ["PROJ_LIB"] = r"C:\Users\yhon\Anaconda3\Lib\site-packages\pyproj\proj_dir\share"; #path of the proj for basemap, not useful on HPC
# from mpl_toolkits.basemap import Basemap

# import pyproj


# import nctoolkit    # https://nctoolkit.readthedocs.io/en/latest/
#%%
# ==================================================================================
# ===============================USER CONTROLS======================================
# ==================================================================================  
LMHOFS_nc=r'C:\FVCOM_Mich\FVCOM_runs\April_2020_lmhofs.nc'
extend_nc=os.path.join(r'C:\FVCOM_Mich\FVCOM_runs\test_lmhofs_extend\run_extend_2','lmhofs_extend_ZhWet.nc')

path_Nodes = r'C:\FVCOM_Mich\Post_Scripts\Nodes_WL'
TWL_csv = r'C:\FVCOM_Mich\Post_Scripts\Nodes_WL\TWL'
df_twl = pd.read_csv(os.path.join(path_Nodes,'TWL\\TWL_FloodTS_GreenBay.csv'), sep=',',index_col=0, parse_dates=True)    # saved in meter, need to convert to ha

# option for saving directory
path_file = r'C:\FVCOM_Mich\Post_Scripts\Flood_comparisons'
Outfig=r'C:\FVCOM_Mich\FVCOM_runs\Fig_extend\Fig_TimeSerie\All'

#%% the node id used for lmhofs water level, node id of 2dm begins with 1, hence 163 here = id 164 in 2dm
list_extend=['greenbay','holland','ludington','montague','muskegon','whiting']
id_node=pd.DataFrame(index=list_extend,columns=['lmhofs', 'extend','Diff_elev']) # Diff_elev is the difference between lmhofs and extent at the same location, finally it is not used

id_node.loc['greenbay','lmhofs'] = 163
id_node.loc['holland','lmhofs'] = 9635
id_node.loc['ludington','lmhofs'] = 22755
id_node.loc['montague','lmhofs'] = 16499
id_node.loc['muskegon','lmhofs'] = 14965
id_node.loc['whiting','lmhofs'] = 2163

id_node.loc['greenbay','extend'] = 74789
id_node.loc['holland','extend'] = 98134
id_node.loc['ludington','extend'] = 135598
id_node.loc['montague','extend'] = 112422
id_node.loc['muskegon','extend'] = 101678
id_node.loc['whiting','extend'] = 96019

#%%
nc_lmhofs= Dataset(LMHOFS_nc, 'r')   # Times begins at 2020-04-01,01:00:00, ends 05-02, 00:00
h_mh=nc_lmhofs.variables['h'][:]
zeta_mh=nc_lmhofs.variables['zeta'][:] 
WL_mh=h_mh+zeta_mh

nc_extend = Dataset(extend_nc, 'r')   # Times begins at 2020-04-01,00:00:00, ends 05-01, 00:00
h_extend=nc_extend.variables['h'][:]
zeta_extend=nc_extend.variables['zeta'][:] 
WL_extend=h_extend+zeta_extend

#%% Calculate the differences of h, zeta, h+zeta for lmhofs and extended grids
ratio_EXT_BTM_TWL = {
    'greenbay': ('GreenBay',1.5,1,10),
    'holland': ('Holland',2,1,0),
    'ludington': ('Ludington',5,1,0.3),
    'montague': ('Montague',2,1,0),
    'muskegon': ('Muskegon',1.8,1,1),
    'whiting': ('Southern-Chicago',5,1,0.2)
}

df_max = pd.DataFrame(index=list_extend, columns=['BTM','EXT','TWL'])

for extend_area in list_extend:
    #%     
    h_lmhofs=h_mh[int(id_node.loc[extend_area,'lmhofs'])]
    zeta_lmhofs=zeta_mh[:,int(id_node.loc[extend_area,'lmhofs'])]
    WL_lmhofs=WL_mh[:,int(id_node.loc[extend_area,'lmhofs'])]    
    # not useful finally, but can keep for other use
    # h_ext=h_extend[int(id_node.loc[extend_area,'extend'])]
    # zeta_ext=zeta_extend[:,int(id_node.loc[extend_area,'extend'])]
    # WL_ext=WL_extend[:,int(id_node.loc[extend_area,'extend'])]        
    
    # fig,(ax1, ax2) = plt.subplots(2,1)
    # ax1.plot(zeta_lmhofs,linestyle=':',color='r',lw=2)
    # ax1.plot(zeta_ext,linestyle='-',color='k',lw=2)
    # ax2.plot(WL_lmhofs,linestyle=':',color='r',lw=2)
    # ax2.plot(WL_ext,linestyle='-',color='k',lw=2)
    # print('h_diff:',h_ext-h_lmhofs)

    # diff=np.mean(WL_ext)-np.mean(WL_lmhofs)   # diff=extent-lmhofs; 
    # print('mean_diff:+',diff)
    # id_node.loc[extend_area,'Diff_elev'] = diff
    
    #% Now start the flood calculation
    file_nc=os.path.join(r'C:\FVCOM_Mich\FVCOM_runs\test_lmhofs_extend\run_extend_2','extend_'+extend_area+'.nc')
    ncfile_small = Dataset(file_nc, 'r')
    h_small=ncfile_small.variables['h'][:]       #(132408)
    zeta_small=ncfile_small.variables['zeta'][:]   # zeta=zeta+1.3 for initiation
    area=ncfile_small.variables['art1'][:]
    wet=ncfile_small.variables['wet_nodes'][:]
    WL_small=h_small+zeta_small
    # time 1 is the initial wet/dry nodes, as restart from time1, add 1.3m
    ref_wet=sum(wet[1]*area)/1000000
    ref_dry=sum((1-wet[1])*area)/1000000
    # np.count_nonzero(h_small==0) # to verify if any h_small==0, returns 0, meaning no 0 h small
    h_dry_ini=(1-wet[1])*h_small   # the h for initially dry nodes
    # zeta_dry_ini=(1-wet[1])*zeta_small[1]   # the h for initially dry nodes
    # WL_ini=(1-wet[1])*WL_small[1]   # the h for initially dry nodes    
    
    # lmhofs
    lmhofs_ini=WL_lmhofs[0]


    begin_time=datetime.datetime(2020, 4, 28, 0)    # plot figures from 04/27
    diff_extend=begin_time-datetime.datetime(2020, 4, 1, 0)
    begin_extend_id=diff_extend.days*24+1                 # hourly outputs
    begin_lmhofs_id=begin_extend_id-1
    end_extend_id=wet.shape[0]
    end_lmhofs_id=end_extend_id-1 

    sim_time=end_lmhofs_id-1-begin_lmhofs_id+1
    flood_area_lmhofs=np.zeros(sim_time)  
    max_flood_lmhofs=np.zeros(sim_time)
    lmhofs_change=np.zeros(sim_time)
    time_list = [begin_time + datetime.timedelta(hours=t) for t in range(sim_time)]    
    
    #% the idea is to calculate the change of lmhofs water level and compare that to h
    for it in range(begin_lmhofs_id,end_lmhofs_id):
        WL_change=WL_lmhofs[it]-lmhofs_ini       
        lmhofs_change[it-begin_lmhofs_id]=WL_change
        flood_mask=np.where(((h_dry_ini<0) & (h_dry_ini>=-1*(WL_change+1.3))), 1, 0)
        # np.count_nonzero(flood_mask==1)        
        flood_area_lmhofs[it-begin_lmhofs_id]=sum(flood_mask*area)/1000000
        max_flood_lmhofs[it-begin_lmhofs_id]=max(flood_mask*(WL_change+1.3+h_dry_ini))        

    #% extended grids
    wet_area_extend=np.zeros(sim_time)
    flood_area_extend=np.zeros(sim_time)
    max_flood_extend=np.zeros(sim_time)
 
    #% evolution of dry and wet areas
    for it in range(begin_extend_id,end_extend_id):
        wet_it=sum(wet[it]*area)/1000000  # convert to km2
        dry_it=sum((1-wet[it])*area)/1000000
        wet_area_extend[it-begin_extend_id]=wet_it-ref_wet
        flood_area_extend[it-begin_extend_id]=ref_dry-dry_it

        wet_mask=wet[it]*(1-wet[1])  # initially not flooded, but laterly flooded
        water_flood=WL_small[it]*wet_mask
        max_flood_extend[it-begin_extend_id]=max(water_flood)
    
    flood_area_extend[flood_area_extend < 0] = 0
    flood_area_lmhofs[flood_area_lmhofs < 0] = 0
    #%

    area_ext = ratio_EXT_BTM_TWL[extend_area][1]*flood_area_extend*100
    area_btm = ratio_EXT_BTM_TWL[extend_area][2]*flood_area_lmhofs*100
    area_twl = ratio_EXT_BTM_TWL[extend_area][3]*df_twl['Sum_Area']*0.002
 
    df_max.loc[extend_area,'BTM'] = np.nanmax(area_btm) 
    df_max.loc[extend_area,'EXT'] = np.nanmax(area_ext)
    df_max.loc[extend_area,'TWL'] = np.nanmax(area_twl) 
 
    fig1=plt.figure(figsize=(10.0,6.0))
    ax1=plt.gca()
    
    ax1.plot(time_list, area_ext, label='EXT',linestyle='-',color='b',lw=2)   # km2 to hectare, if added wave, use '-.' as line style
    ax1.plot(time_list, area_btm, label='BTM', linestyle='-.',color='k',lw=2)   # km2 to hectare
    ax1.plot(df_twl.index,area_twl, label='TWL', linestyle='--',color='g',lw=2)   
  
    ax1.set_xlim(time_list[0], time_list[-1])
    ax1.set_ylabel('Flooding area (ha)',fontsize=14, weight='bold')
    ax1.set_xlabel('Date',fontsize=14, weight='bold')
    ax1.legend(loc='upper left',fontsize=14, ncol=1).set_title('')
    
    plt.title('Flooding area for '+ ratio_EXT_BTM_TWL[extend_area][0],fontsize=16, weight='bold')
    ax1.tick_params(axis='both', labelsize=14)
    
    xfmt = md.DateFormatter('%m-%d')
    ax1.xaxis.set_major_formatter(xfmt)
    
    fig_name1=os.path.join(Outfig,'Flood_Comp_'+extend_area+'.jpg')
    plt.savefig(fig_name1, dpi=300)
 
df_max.to_csv(os.path.join(path_file,'Max_Area.csv'), index=True)    




# #%%
#     # =============================================================================
#     # maximum flood water depth
#     # =============================================================================
 
#     fig2=plt.figure(figsize=(10.0,6.0))
#     ax2=plt.gca()
#     plt.plot(time_list, max_flood_extend,label='Hydrodynamic',linestyle='-',color='b',lw=2)   # meter
#     plt.plot(time_list, max_flood_lmhofs, label='Bathtub', linestyle='-',color='k',lw=2)   # km2 to hectare
    
#     ax2.set_ylabel('Maximum Flooding Depth (m)',fontsize=14, weight='bold')
#     ax2.set_xlabel('Date',fontsize=14, weight='bold')
#     plt.legend()
#     ax2.legend(loc='upper left',fontsize=14, ncol=1).set_title('')
    
#     plt.title('Maximum Flooding Depth at '+extend_area,fontsize=16, weight='bold')
#     ax2.tick_params(axis='both', labelsize=14)
    
#     xfmt = md.DateFormatter('%m-%d')
#     ax2.xaxis.set_major_formatter(xfmt)
    
#     fig_name2=os.path.join(Outfig,'Flood_depth_Comp_'+extend_area+'.jpg')
#     plt.savefig(fig_name2, dpi=300)
    
# #% changes of lmhofs
#     fig3=plt.figure(figsize=(10.0,6.0))
#     ax3=plt.gca()
#     plt.plot(time_list, lmhofs_change) 
#     ax3.set_ylabel('Water Level Changes (m)',fontsize=14, weight='bold')
#     ax3.set_xlabel('Date',fontsize=14, weight='bold')   
#     plt.title('LMHOFS water level changes for '+extend_area,fontsize=16, weight='bold')
#     ax3.tick_params(axis='both', labelsize=14)
    
#     xfmt = md.DateFormatter('%m-%d')
#     ax3.xaxis.set_major_formatter(xfmt)
    
#     fig_name3=os.path.join(Outfig,'Lmhofs_WL_'+extend_area+'.jpg')
#     plt.savefig(fig_name3, dpi=300)    
    
    
    
    