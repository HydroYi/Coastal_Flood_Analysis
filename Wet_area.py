# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# ** Calculate and plot dry/wet areas of fvcom output
# ** R scripts can also be used to plot maps
# ** A bash script is available for downloading data
# ** Edit by Yi Hong, 04/08/2021
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
os.environ["PROJ_LIB"] = r"C:\Users\yhon\Anaconda3\Lib\site-packages\pyproj\proj_dir\share"; #path of the proj for basemap, not useful on HPC
from mpl_toolkits.basemap import Basemap

import pyproj


# import nctoolkit    # https://nctoolkit.readthedocs.io/en/latest/
#%%
# ==================================================================================
# ===============================USER CONTROLS======================================
# ==================================================================================  
# SET file to read in
# test_nc=os.path.join(r'C:\FVCOM_Mich\FVCOM_runs\test_lmhofs_extend\run_extend_2','extend_muskegon.nc')
# option to save plot or show on screen:

# option for saving directory
Outfig=r'C:\FVCOM_Mich\FVCOM_runs\Fig_extend\Fig_TimeSerie'
# OutDir = os.path.join(r'C:\FVCOM_Mich\FVCOM_runs\test_real5_1', 'output')          # Output Directory

# ncfile = Dataset(test_nc, 'r')
# print(ncfile.variables.keys())  
# ncfile4 = Dataset(test_nc, 'r')
# print(ncfile4.variables.keys())  
# #%%
# # nv=ncfile4.variables['nv'][:]-1  # node surronding element, shape (3,249069)
# # nvt=nv.transpose()
# lat=ncfile.variables['lat'][:]
# lon=ncfile.variables['lon'][:]   #
# # latc=ncfile4.variables['latc'][:]  # zonal lat (249069,)
# # lonc=ncfile4.variables['lonc'][:]  # zonal lon
# # T=ncfile.variables['temp'][t0,z0,:]
# h=ncfile.variables['h'][:]       #(132408)
# # h_center=ncfile.variables['h_center'][:]  # (249069)
# zeta=ncfile.variables['zeta'][:] 
# area=ncfile.variables['art1'][:]
# wet=ncfile.variables['wet_nodes'][:]
# temp=ncfile.variables['temp'][:]
# # uwind=ncfile.variables['uwind_speed'][:] 
# # vwind=ncfile.variables['vwind_speed'][:] 
# # Times=ncfile.variables['Times'][t0]
# # P=ncfile.variables['atmos_press'][:]
# water_level=h+zeta
# end_time=zeta.shape[0]

# #%% test the simulation area

# fig,ax=plt.subplots(figsize=(16,10))
# ax.plot(lon, lat, 'o')
# plt.title('Extend muskegon')
# save_name=os.path.join(Outfig,'Extend_muskegon.jpg')
# plt.savefig(save_name, dpi=300)

# #%% test Distribution of art1 in time 0, time 1
# max(area);min(area)   # the unit of area is meter
# plt.hist(area)
#%%
# =============================================================================
# Set a loop to plot for each extend area
# =============================================================================
for extend_area in ['greenbay','holland','ludington','montague','muskegon','whiting']:
# for extend_area in ['holland','ludington','montague','whiting']:
        #%
    file_nc=os.path.join(r'C:\FVCOM_Mich\FVCOM_runs\test_lmhofs_extend\run_extend_2','extend_'+extend_area+'.nc')
    ncfile = Dataset(file_nc, 'r')
    h=ncfile.variables['h'][:]       #(132408)
    zeta=ncfile.variables['zeta'][:] 
    area=ncfile.variables['art1'][:]
    wet=ncfile.variables['wet_nodes'][:]
    water_level=h+zeta
    end_time_id=zeta.shape[0]
    # lat=ncfile.variables['lat'][:]
    # lon=ncfile.variables['lon'][:]   #
    # plt.plot(lon, lat, 'o')
    #% set time 0 as the reference, wet
    ref_wet=sum(wet[1]*area)/1000000
    ref_dry=sum((1-wet[1])*area)/1000000
    #%
    # ref_wet=sum(wet[0]*area)/1000000
    # ref_dry=sum((1-wet[0])*area)/1000000
    begin_time=datetime.datetime(2020, 4, 28, 0)    # plot figures from 04/27
    diff=begin_time-datetime.datetime(2020, 4, 1, 0)
    begin_time_id=diff.days*24+1                 # hourly outputs
    
    sim_time=end_time_id-1-begin_time_id+1
    wet_area=np.zeros(sim_time)
    flood_area=np.zeros(sim_time)
    
    time_list = [begin_time + datetime.timedelta(hours=t) for t in range(sim_time)]
    
    #% evolution of dry and wet areas
    for it in range(begin_time_id,end_time_id):
        wet_it=sum(wet[it]*area)/1000000  # convert to km2
        dry_it=sum((1-wet[it])*area)/1000000
        wet_area[it-begin_time_id]=wet_it-ref_wet
        if ref_dry > dry_it:
            flood_area[it-begin_time_id]=ref_dry-dry_it
        else:
            flood_area[it-begin_time_id]=0
    
    #% plot
    # flood_area=-1*dry_area/1000000 
    # wet_area=wet_area/1000000
    
    fig=plt.figure(figsize=(10.0,6.0))
    ax=plt.gca()
    # plt.plot(time_list, flood_area*247.11)   # km2 to acres
    plt.plot(time_list, flood_area*100)   # km2 to hectare
    ax.set_ylabel('Flooding area (ha)',fontsize=14, weight='bold')
    ax.set_xlabel('Date',fontsize=14, weight='bold')
    plt.title('Flooding area for '+extend_area+' Harbor',fontsize=16, weight='bold')
    ax.tick_params(axis='both', labelsize=14)
    
    xfmt = md.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    
    fig_name=os.path.join(Outfig,'Flood_'+extend_area+'.jpg')
    plt.savefig(fig_name, dpi=300)
    
    # #% 
    # =============================================================================
    # maximum flood water depth
    # =============================================================================
    max_flood=np.zeros(sim_time)
    for it in range(begin_time_id,end_time_id):
        wet_mask=wet[it]*(1-wet[1])  # initially not flooded, but laterly flooded
        water_flood=water_level[it]*wet_mask
        max_flood[it-begin_time_id]=max(water_flood)
    
    #% plot
    
    fig=plt.figure(figsize=(10.0,6.0))
    ax=plt.gca()
    plt.plot(time_list, max_flood)   # meter
    ax.set_ylabel('Maximum Flooding Depth (m)',fontsize=14, weight='bold')
    ax.set_xlabel('Date',fontsize=14, weight='bold')
    plt.title('Maximum Flooding Depth at '+extend_area,fontsize=16, weight='bold')
    ax.tick_params(axis='both', labelsize=14)
    
    xfmt = md.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(xfmt)
    
    fig_name=os.path.join(Outfig,'Flood_depth_'+extend_area+'.jpg')
    plt.savefig(fig_name, dpi=300)
    
