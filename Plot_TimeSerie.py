# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# ** Test the FVCOM plots
# ** Use this script to plot timeseries, 
# ** R scripts are used to plot maps
# ** A bash script is available for downloading data
# ** Edit by Yi Hong, 12/17/2020
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
'''
Plot maps with PyFVCOM package
Guage locations:
    Holland: including water level and wave, IGLD datatum (degree and decimal minutes)
    Latitude	42° 46.4 N = 42.773333
    Longitude	86° 12.8 W = - 86.213333

    Ludington, MI
    Latitude	43° 56.8 N = 43.9466667
    Longitude	86° 26.5 W = -86.4416667   
    
    Calumet Harbor (Chicago), IL - Station ID: 9087044
    Latitude	41° 43.8 N = 41.73
    Longitude	87° 32.3 W = 87.5383333

    '''
    
import os
# import xarray as xr
import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from math import cos, asin, sqrt
import datetime

# os.environ["PROJ_LIB"] = r"C:\Users\yhon\Anaconda3\Library\share"; #path of the proj for basemap, not useful on HPC
# from mpl_toolkits.basemap import Basemap

#%%
# ==================================================================================
# ===============================USER CONTROLS======================================
# ==================================================================================

#set time idx to plot 
t0=0

# set vertical level to plot (0 is surface)
z0=0    

# SET file to read in
OutDir = os.path.join(r'C:\FVCOM_Mich\FVCOM_runs\test_real5_1', 'output')          # Output Directory
# test_nc=os.path.join(OutDir,'out_z.nc')
test_nc=os.path.join(OutDir,'Merged_6_0001_ZhP.nc')
gauge_dir=r'C:\FVCOM_Mich\Data\Gauge_data\20200401_0501'
meteo_dir=r'C:\FVCOM_Mich\Data\Gauge_data\Meteo'
# +74.2, IGLD, Meters


# option to save plot or show on screen:
save=1  # 1=save plot; 0=draw plot

# option for figure saving directory
Outfig=os.path.join(OutDir,'Figs')

#%%
# ==================================================================================
# ==================================================================================
# ==================================================================================

#print ('reading file...')# python 3 syntax
#print 'reading file...' # python 2 syntax
ncfile = Dataset(test_nc, 'r')
print(ncfile.variables.keys())  
#%%
nv=ncfile.variables['nv'][:]-1  # node surronding element, shape (3,249069)
nvt=nv.transpose()
lat=ncfile.variables['lat'][:]
lon=ncfile.variables['lon'][:]   #
latc=ncfile.variables['latc'][:]  # zonal lat (249069,)
lonc=ncfile.variables['lonc'][:]  # zonal lon
# T=ncfile.variables['temp'][t0,z0,:]
h=ncfile.variables['h'][:]       #(132408)
h_center=ncfile.variables['h_center'][:]  # (249069)
zeta=ncfile.variables['zeta'][:] 
uwind=ncfile.variables['uwind_speed'][:] 
vwind=ncfile.variables['vwind_speed'][:] 
Times=ncfile.variables['Times'][t0]

water_level=h+zeta

#%% Plot time series comparing to the gauge observations
# Function to calculat the distance between two points with given lat and lon
# =============================================================================
# The Haversine formula is needed for a correct calculation of the distance between points on the globe
    
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Math.PI / 180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))  #2 * R; R = 6371 km

# Function to define the closest index of the nodes to gauge stations
def closest_index(list_coord, point_coord):
    min_dist=min(list_coord, key=lambda point: distance(point_coord['Lat'],point_coord['Lon'],point['Lat'],point['Lon']))
    return list_coord.index(min_dist)
#%%
loc_gauge=pd.DataFrame(index=['holland','ludington'],columns=['Lat', 'Lon'])
loc_gauge.loc['holland']={'Lat':42.773333,'Lon':-86.213333+360}  # lon in FVCOM is E
loc_gauge.loc['ludington']={'Lat':43.9466667,'Lon':-86.4416667+360 }
# #%
list_Lat=lat.tolist()
list_Lon=lon.tolist()
list_dict_coord = [{'Lat':list_Lat[i], 'Lon':list_Lon[i]} for i in range(len(list_Lon))]
#%
loc_gauge.loc['holland','Index_node']=closest_index(list_dict_coord,loc_gauge.loc['holland'])  # 83951
loc_gauge.loc['ludington','Index_node']=closest_index(list_dict_coord,loc_gauge.loc['ludington'])  # 91777
# to avoid ocsillations in the harbor, select a inlake point
loc_gauge.loc['holland','Index2']=91858
loc_gauge.loc['ludington','Index2']=122340
#%% Get the water level at the gauge point
begin_time=datetime.datetime(2020, 4, 1, 0)
time_list = [begin_time + datetime.timedelta(hours=t) for t in range(water_level.shape[0])]

#%%
for gauge in ['holland','ludington']:
    index=loc_gauge.loc[gauge,'Index_node']    
    height=water_level[:,int(index)]
    index2=loc_gauge.loc[gauge,'Index2']    
    height2=water_level[:,int(index2)]
#     gauge_file=os.path.join(gauge_dir,gauge+'.csv')
#     obs_height=pd.read_csv(gauge_file,delimiter=',')
#     obs_height['Date Time']=pd.to_datetime(obs_height['Date Time'])
#     obs_height=obs_height.sort_values(by=['Date Time']).set_index('Date Time')
# #%
#     fig1, ax = plt.subplots(1)
#     obs_height[' Water Level'].plot(ax=ax,linestyle=':',color='r',lw=4)
#     fig_name=os.path.join(Outfig,'Observed_'+gauge+'.jpg')
#     plt.title('Obeserved water level at '+ gauge)
#     fig1.savefig(fig_name, dpi=300)
    
    fig2, ax2 = plt.subplots(1)
    plt.plot(time_list, height)
    fig2_name=os.path.join(Outfig,'FVCOM_'+gauge+'.jpg')
    plt.title('FVCOM water level at '+ gauge)
    fig2.savefig(fig2_name, dpi=300)

    fig3, ax3 = plt.subplots(1)
    plt.plot(time_list, height2)
    fig3_name=os.path.join(Outfig,'FVCOM_'+gauge+'_out.jpg')
    plt.title('FVCOM water level at outside '+ gauge)
    fig3.savefig(fig3_name, dpi=300)
#%%
    fig, ax = plt.subplots(1)
    obs_height[' Water Level'].plot(ax=ax,linestyle=':',color='r',lw=4)
    ax.plot(time_list, height,linestyle='-',color='b',lw=4)
    ax.set(xlabel='Time',ylabel='Water level (m)')
    ax.legend(labels=['Observation','FVCOM'],loc='upper right')
    # ax.xaxis.set_major_locator(mdates.MonthLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    #% ax.tick_params(axis='x', rotation=70)
    plt.show()
    fig_name=os.path.join(Outfig,'TimeSeries '+gauge+'.jpg')
    plt.title('Water level at '+ gauge)
    fig.savefig(fig_name, dpi=300)

#%%
# =============================================================================
# Compare to the lmhofs data
# =============================================================================
lmhofs_nc=os.path.join(OutDir,'lmhofs_april.nc')
ncfile2 = Dataset(lmhofs_nc, 'r')
print(ncfile2.variables.keys())
#%%
lat2=ncfile2.variables['lat'][:]
lon2=ncfile2.variables['lon'][:]   #
# T=ncfile.variables['temp'][t0,z0,:]
h2=ncfile2.variables['h'][:]       #(132408)
zeta2=ncfile2.variables['zeta'][:] 
# Times2=ncfile2.variables['Times'][t0]

water_level2=h2+zeta2
#%%
list_Lat2=lat2.tolist()
list_Lon2=lon2.tolist()
list_dict_coord2 = [{'Lat':list_Lat2[i], 'Lon':list_Lon2[i]} for i in range(len(list_Lon2))]
#%
loc_gauge.loc['holland','Index_lmhofs']=closest_index(list_dict_coord2,loc_gauge.loc['holland'])
loc_gauge.loc['ludington','Index_lmhofs']=closest_index(list_dict_coord2,loc_gauge.loc['ludington'])

#%%
begin_time=datetime.datetime(2020, 4, 1, 0)
time_list = [begin_time + datetime.timedelta(hours=t) for t in range(water_level.shape[0])]
time_list3 = [begin_time + datetime.timedelta(hours=t) for t in range(water_level2.shape[0])]


for gauge in ['holland','ludington']:
    #%
    index=loc_gauge.loc[gauge,'Index_node']    
    height=water_level[:,int(index)]
    index2=loc_gauge.loc[gauge,'Index2']    
    height2=water_level[:,int(index2)]
    index3=loc_gauge.loc[gauge,'Index_lmhofs']    
    height3=water_level2[:,int(index3)]
#     gauge_file=os.path.join(gauge_dir,gauge+'.csv')
#     obs_height=pd.read_csv(gauge_file,delimiter=',')
#     obs_height['Date Time']=pd.to_datetime(obs_height['Date Time'])
#     obs_height=obs_height.sort_values(by=['Date Time']).set_index('Date Time')
# #%
#     fig1, ax = plt.subplots(1)
#     obs_height[' Water Level'].plot(ax=ax,linestyle=':',color='r',lw=4)
#     fig_name=os.path.join(Outfig,'Observed_'+gauge+'.jpg')
#     plt.title('Obeserved water level at '+ gauge)
#     fig1.savefig(fig_name, dpi=300)
    
    # fig2, ax2 = plt.subplots(2)
    # plt.plot(time_list, height,time_list, height2)
    # fig2_name=os.path.join(Outfig,'FVCOM_'+gauge+'.jpg')
    # plt.title('FVCOM3 water level at '+ gauge)
    # ax2.legend(labels=['Harbor','Lake'])
    # fig2.savefig(fig2_name, dpi=300)

    fig3, ax3 = plt.subplots(1)
    plt.plot(time_list, height3[0:577])
    fig3_name=os.path.join(Outfig,'LMHOFS_'+gauge+'.jpg')
    plt.title('LMHOFS water level at '+ gauge)
    # ax3.legend(labels=['Harbor','Lake'])
    fig3.savefig(fig3_name, dpi=300)
    
#%% Test a nodes in the middle of the lake    
index=125858   
height=water_level[:,int(index)]
fig1, ax = plt.subplots(1)
plt.plot(time_list, height)
fig_name=os.path.join(Outfig,'FVCOM_125859.jpg')
plt.title('FVCOM water level of the middle lake (node 125859)')
fig1.savefig(fig_name, dpi=300)

#%% test h, zeta, h_center, lundington
index1=91777  # h_index1=6.1342626
index2=122340 # h_index2=9.12322
  
zeta_lu=zeta[:,int(index2)]

gauge_file=os.path.join(gauge_dir,'ludington.csv')
obs_height=pd.read_csv(gauge_file,delimiter=',')
obs_height['Date Time']=pd.to_datetime(obs_height['Date Time'])
obs_height=obs_height.sort_values(by=['Date Time']).set_index('Date Time')
#%
fig2, ax2 = plt.subplots(1)

obs_height[' Water Level'].plot(ax=ax2,linestyle=':',color='r',lw=2)
plt.plot(time_list, zeta_lu+177.35,color='b',lw=2)

fig_name=os.path.join(Outfig,'water_Ludington2.jpg')
plt.title('Water level at Ludington out (zeta+177.35)')
fig2.savefig(fig_name, dpi=300)





#%%
# =============================================================================
# test wind
# =============================================================================
# holland: 156832, out: 171742
# ludington: 171597, out: 229471

meteo_file=os.path.join(meteo_dir,'Ludington_met.csv')
obs_meteo=pd.read_csv(meteo_file,delimiter=',')
obs_meteo['Date_Time'] = pd.to_datetime(obs_meteo['Date'] + ' ' + obs_meteo['Time (LST/LDT)'])
obs_meteo=obs_meteo.sort_values(by=['Date_Time']).set_index('Date_Time')
obs_meteo['Wind Speed (m/s)']=obs_meteo['Wind Speed (m/s)'].apply(pd.to_numeric, args=('coerce',))

u_wind=uwind[:,171597] 
v_wind=vwind[:,171597]
wind_speed=np.sqrt(u_wind**2+v_wind**2)
#%%
fig1,ax1 = plt.subplots(1)
plt.plot(time_list, wind_speed,color='b',lw=3)
obs_meteo['Wind Speed (m/s)'].plot(ax=ax1,linestyle=':',color='r',lw=3)

plt.title('Wind speed (m/s) at Ludingvig')
fig_name=os.path.join(Outfig,'Wind_speed_ludingvig.jpg')
fig1.savefig(fig_name, dpi=300)


#%%









