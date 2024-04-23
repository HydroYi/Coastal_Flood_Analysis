# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
# ** Test the FVCOM plots
# ** Use this script to plot timeseries, 
# ** R scripts are used to plot maps
# ** A bash script is available for downloading data
# ** Edit by Yi Hong, 12/17/2020
# ** Modified 10/15/2022, observation to 1 hour time step for smooth plot
# ** Modified 11/19/2022, add RMSE and NSE function between observation and simulation
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
'''
Plot maps with PyFVCOM package
Guage locations:
    Holland: including water level and wave, ID: 9087031
    Latitude	42° 46.4 N = 42.773333
    Longitude	86° 12.8 W = - 86.213333

    Ludington, MI, ID: 9087023
    Latitude	43° 56.8 N = 43.9466667
    Longitude	86° 26.5 W = -86.4416667   
    
    Calumet Harbor (Whiting), IL - Station ID: 9087044
    Latitude	41° 43.8 N = 41.73
    Longitude	87° 32.3 W = 87.5383333

    Green Bay (East), WI - Station ID: 9087077
    Latitude	44° 32.3 N = 44.5383333
    Longitude	88° 0.1 W = 88.0016667

    '''
    
import os
# import xarray as xr
import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from math import cos, asin, sqrt
import datetime
from matplotlib.dates import DateFormatter

# os.environ["PROJ_LIB"] = r"C:\Users\yhon\Anaconda3\Library\share"; #path of the proj for basemap, not useful on HPC
# from mpl_toolkits.basemap import Basemap

#%% Function to calculat the distance between two points with given lat and lon
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

# location of Holland, Ludington gauge stations
loc_gauge=pd.DataFrame(index=['holland','ludington','whiting','greenbay'],columns=['Lat', 'Lon'])
loc_gauge.loc['holland']={'Lat':42.773333,'Lon':-86.213333+360}  # lon in FVCOM is E
loc_gauge.loc['ludington']={'Lat':43.9466667,'Lon':-86.4416667+360}
loc_gauge.loc['whiting']={'Lat':41.73,'Lon':-87.5383333+360}
loc_gauge.loc['greenbay']={'Lat':44.5383333,'Lon':-88.0016667+360}
loc_gauge.to_csv('C:\FVCOM_Mich\Data\Gauge_data\Gauge_locations.csv', sep=',', index=False)

#%%
# =============================================================================
# Functions to calculate the RMSE and MAE, and (prod-obs)/obs
# =============================================================================
def MAE_RMSE_Diff(serie1, serie2):
    if len(serie1)==len(serie2):
        serie1=serie1.reset_index(drop=True)
        serie2=serie2.reset_index(drop=True)
        
        MAE=(serie1-serie2).abs().mean()
        RMSE=np.sqrt(((serie1-serie2)**2).mean())
        Diff=(serie1-serie2).sum(min_count=1)
        return MAE, RMSE, Diff
    else:
        raise Exception('Length of the two series are not equal')
        return []

def R2(serie1, serie2):  # this function is to calculate the determinant R2
    if len(serie1)==len(serie2):
        if serie1.isnull().all() or serie2.isnull().all():
            return np.nan
        else:            # Compute only when both S1 and S2 is true
            S1=[]
            S2=[]            
            for i in range(len(serie1)):
                if (not np.isnan(serie1.iloc[i]) and not np.isnan(serie2.iloc[i])):
                    S1.append(serie1.iloc[i])
                    S2.append(serie2.iloc[i])       
            if len(S1)==0:
                return np.nan
            else:
                mean_S1=sum(S1)/len(S1)
                mean_S2=sum(S2)/len(S2)
                diff_S12=0
                diff2_S1=0
                diff2_S2=0  
                for j in range(len(S1)):
                    diff_S12 += (S1[j]-mean_S1)*(S2[j]-mean_S2)
                    diff2_S1 += (S1[j]-mean_S1)**2
                    diff2_S2 += (S2[j]-mean_S2)**2

                if diff2_S1 != 0 and diff2_S2 != 0:
                    R2 = diff_S12/(np.sqrt(diff2_S1)*np.sqrt(diff2_S2))
                    return R2
                else:
                    return np.nan
    else:
        raise Exception('Length of the two series are not equal')
        return np.nan  

#%%
# ==================================================================================
# ===============================USER CONTROLS======================================
# ==================================================================================

#set time idx to plot 
t0=0

# set vertical level to plot (0 is surface)
z0=0    

# SET file to read in
# OutDir1 = os.path.join(r'C:\FVCOM_Mich\FVCOM_runs\test_real5_1', 'output') 
# OutDir2 = os.path.join(r'C:\FVCOM_Mich\FVCOM_runs\test_real5_2', 'output')          # Output Directory
# test_nc=os.path.join(r'C:\FVCOM_Mich\FVCOM_runs\test_lmhofs_extend\run1','lmhofs_extend_zh.nc')
test_nc1=os.path.join(r'C:\FVCOM_Mich\FVCOM_runs\test_lmhofs_extend\run_extend_2','lmhofs_extend_ZhWet.nc')
# test_nc2=os.path.join(r'C:\FVCOM_Mich\FVCOM_runs\test_lmhofs_extend\run1','lmhofs_extend_zh2.nc')

LMHOFS_nc=r'C:\FVCOM_Mich\FVCOM_runs\April_2020_lmhofs.nc'
gauge_dir=r'C:\FVCOM_Mich\Data\Gauge_data\20200401_0501'
meteo_dir=r'C:\FVCOM_Mich\Data\Gauge_data\Meteo'
# +74.2, IGLD, Meters
HRRR_dir=r'C:\FVCOM_Mich\Data\HRRR'
# option to save plot or show on screen:
save=1  # 1=save plot; 0=draw plot

# option for figure saving directory
Outfig=r'C:\FVCOM_Mich\FVCOM_runs\Figs'

#%%
# ==================================================================================
# ==================================================================================
# ==================================================================================

#print ('reading file...')# python 3 syntax
#print 'reading file...' # python 2 syntax
ncfile1 = Dataset(test_nc1, 'r')
# ncfile2 = Dataset(test_nc2, 'r')
# ncfile = Dataset(test_nc, 'r')
nc_lmhofs= Dataset(LMHOFS_nc, 'r')

# print(ncfile.variables.keys())  
print(ncfile1.variables.keys())  
print(nc_lmhofs.variables.keys())  

#%% Check start and end time
start_nc1=datetime.datetime(1970, 1, 1, 0, 0)  + datetime.timedelta(float(ncfile1.variables['time'][0].data))
end_nc1=datetime.datetime(1970, 1, 1, 0, 0)  + datetime.timedelta(float(ncfile1.variables['time'][ncfile1.variables['time'].shape[0]-1].data))
# start_nc2=datetime.datetime(1970, 1, 1, 0, 0)  + datetime.timedelta(float(ncfile2.variables['time'][0].data))
# end_nc2=datetime.datetime(1970, 1, 1, 0, 0)  + datetime.timedelta(float(ncfile2.variables['time'][ncfile2.variables['time'].shape[0]-1].data))

start_nc3=datetime.datetime(1970, 1, 1, 0, 0)  + datetime.timedelta(float(nc_lmhofs.variables['time'][0].data))
end_nc3=datetime.datetime(1970, 1, 1, 0, 0)  + datetime.timedelta(float(nc_lmhofs.variables['time'][nc_lmhofs.variables['time'].shape[0]-1].data))

#%% find the nearest point in LMHOFS

lat_mh=nc_lmhofs.variables['lat'][:]
lon_mh=nc_lmhofs.variables['lon'][:]   #
# T=ncfile.variables['temp'][t0,z0,:]
h_mh=nc_lmhofs.variables['h'][:]       #(132408)
zeta_mh=nc_lmhofs.variables['zeta'][:] 
# Times2=ncfile2.variables['Times'][t0]
#%%
# lat1=ncfile.variables['lat'][:]
# lon1=ncfile.variables['lon'][:]   #
h1=ncfile1.variables['h'][:] 
zeta1=ncfile1.variables['zeta'][:] 
# uwind1=ncfile1.variables['uwind_speed'][:] 
# vwind1=ncfile1.variables['vwind_speed'][:] 

# h2=ncfile2.variables['h'][:] 
# zeta2=ncfile2.variables['zeta'][:] 
# uwind2=ncfile2.variables['uwind_speed'][:] 
# vwind2=ncfile2.variables['vwind_speed'][:] 

water_level1=h1+zeta1
# water_level2=h2+zeta2

water_level_mh=h_mh+zeta_mh
# #%%
# list_Lat=lat1.tolist()
# list_Lon=lon1.tolist()
# list_dict_coord = [{'Lat':list_Lat[i], 'Lon':list_Lon[i]} for i in range(len(list_Lon))]

# Index_holland=closest_index(list_dict_coord,loc_gauge.loc['holland'])    # 9635,98134
# Index_ludington=closest_index(list_dict_coord,loc_gauge.loc['ludington'])  # 22496,95165
# Index_whiting=closest_index(list_dict_coord,loc_gauge.loc['whiting'])    # 9635,91526
# Index_greenbay=closest_index(list_dict_coord,loc_gauge.loc['greenbay'])  # 74404

# #%% index for Grid Merged_6
# loc_gauge.loc['holland','Index_node']= 83951  # closed point
# loc_gauge.loc['ludington','Index_node']= 91777
# # to avoid ocsillations in the harbor, select a inlake point
# loc_gauge.loc['holland','Index2']=91858
# loc_gauge.loc['ludington','Index2']=122340
#%% index for Grid lmhofs extend
loc_gauge.loc['holland','Index_node']= 98134  # closed point
loc_gauge.loc['ludington','Index_node']= 95165
loc_gauge.loc['whiting','Index_node']= 94725
loc_gauge.loc['greenbay','Index_node']= 74789
# to avoid ocsillations in the harbor, select a inlake point
loc_gauge.loc['holland','Index2']=102974
loc_gauge.loc['ludington','Index2']=138497
loc_gauge.loc['whiting','Index2']=101874
loc_gauge.loc['greenbay','Index2']=78973
# lmhofs
loc_gauge.loc['holland','Index_lmhofs'] = 9635
loc_gauge.loc['ludington','Index_lmhofs'] = 22755
loc_gauge.loc['whiting','Index_lmhofs'] = 2163
loc_gauge.loc['greenbay','Index_lmhofs'] = 163
#%%
begin_time=datetime.datetime(2020, 3, 31, 20)
begin_time2=datetime.datetime(2020, 4, 1, 1)
# time_list = [begin_time + datetime.timedelta(hours=t) for t in range(water_level1.shape[0]+water_level2.shape[0]-1)]
time_list = [begin_time + datetime.timedelta(hours=t) for t in range(water_level1.shape[0])]
# time_list2 = [begin_time2 + datetime.timedelta(hours=t) for t in range(water_level1.shape[0]+water_level2.shape[0]-1)]
# time_list2 = [begin_time2 + datetime.timedelta(hours=t) for t in range(water_level1.shape[0])]


#%% plot
myFmt = DateFormatter("%m-%d") # plot xticks format

for gauge in ['holland','ludington','whiting','greenbay']:
    #%
    if gauge =='ludington':
        dh_MH=170.92
        # dh_MH2=dh_MH+1.37    # cold start
        dh_extend=170.25   # +1.3 different initial zeta
        dh_extend_out=164.56
    
    if gauge =='holland':
        dh_MH=174.05
        # dh_MH2=dh_MH+1.37
        dh_extend=169.55
        dh_extend_out=165.46        
                
    if gauge =='whiting':        
        dh_MH=171.88
        # dh_MH2=dh_MH+1.37
        dh_extend=170.05
        dh_extend_out=169.75

    if gauge =='greenbay':  # no data for greenbay during that period
        dh_MH=174.64
        # dh_MH2=dh_MH+1.37
        dh_extend=171.64
        dh_extend_out=174.29
    
    if gauge =='whiting':        
        gauge_name='Southern Chicago'
    else:
        gauge_name = gauge
        
    # height1=water_level1[:,int(loc_gauge.loc[gauge,'Index_lmhofs'])]
    height1=water_level1[:,int(loc_gauge.loc[gauge,'Index_node'])]
    S_h1 = pd.Series(dh_extend+height1[0:len(time_list)])
    # height2=water_level2[:,int(loc_gauge.loc[gauge,'Index_node'] )]
    # Water_extend=np.concatenate((height1, height2[1:height2.shape[0]]))   
    
    height1_out=water_level1[:,int(loc_gauge.loc[gauge,'Index2'])]
    S_h1_out = pd.Series(dh_extend_out+height1_out) 
    # height2_out=water_level2[:,int(loc_gauge.loc[gauge,'Index2'] )]
    # Water2_extend=np.concatenate((height1_out, height2_out[1:height2_out.shape[0]])) 
    
    Water_MH=water_level_mh[:,int(loc_gauge.loc[gauge,'Index_lmhofs'])]
    S_MH = pd.Series(dh_MH+Water_MH[0:len(time_list)])
#% 
    gauge_file=os.path.join(gauge_dir,gauge+'.csv')
    obs_height=pd.read_csv(gauge_file,delimiter=',')
    obs_height['Date Time']=pd.to_datetime(obs_height['Date Time'])
    obs_height=obs_height.sort_values(by=['Date Time']).set_index('Date Time')
    obs_height = obs_height[' Water Level'].resample('H').mean()
    
    rmse_mh = MAE_RMSE_Diff(obs_height, S_MH)[1]
    rmse_h1 = MAE_RMSE_Diff(obs_height, S_h1)[1]    
    rmse_h1_out = MAE_RMSE_Diff(obs_height, S_h1_out)[1]    
    
#%
    fig1,ax1 = plt.subplots(nrows=1, ncols=1,figsize=(8,5))
    # obs_height[' Water Level'].plot(ax=ax1,linestyle=':',color='r',lw=2)
    plt.plot(obs_height.index, obs_height,linestyle=':',color='r',lw=2)
    plt.plot(time_list, S_h1_out,linestyle='-',color='g',lw=1)  
    plt.plot(time_list, S_h1,color='b',lw=1)
    plt.plot(time_list, S_MH,color='k',lw=1)

  
    # plt.plot(time_list2, dh_extend+Water_extend[0:len(time_list2)],color='b',lw=2)
    # plt.plot(time_list2, dh_extend_out+Water2_extend,linestyle='--',color='g',lw=1)
 
    
    ax1.legend(labels=['Observation', 'BTM, RMSE='+"%.2f" %rmse_h1_out, 'EXT, RMSE='+"%.2f" %rmse_h1, 
                       'TWL, RMSE='+"%.2f" %rmse_h1],ncol=1,fontsize=16,title='', loc='upper left')
    ax1.set_ylabel('Water Level (m)')
    ax1.set_xlim(datetime.datetime.strptime('2020401','%Y%m%d'), datetime.datetime.strptime('20200430-23','%Y%m%d-%H'))
    ax1.set_title('Water level validation at '+ gauge_name, fontsize=14, weight='bold') 
    ax1.set_ylabel('Water Level (m)',fontsize=12, weight='bold')
    ax1.set_xlabel('Date',fontsize=12, weight='bold')
    ax1.tick_params(axis='y', labelsize=12)  
    ax1.tick_params(axis='x', labelsize=12)  
    # ax1.set_xticks([])
    ax1.xaxis.set_major_formatter(myFmt)

        
    fig_name=os.path.join(Outfig,'WL_Hour_'+gauge+'_Comp.jpg')
    fig1.savefig(fig_name, dpi=300)

    
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

u_wind1=uwind1[:,171597] 
v_wind1=vwind1[:,171597]
wind1_speed=np.sqrt(u_wind1**2+v_wind1**2)

u_wind2=uwind2[:,171597] 
v_wind2=vwind2[:,171597]
wind2_speed=np.sqrt(u_wind2**2+v_wind2**2)

wind_speed=np.concatenate((wind1_speed, wind2_speed[1:wind2_speed.shape[0]]))

#%%
fig1,ax1 = plt.subplots(1)
plt.plot(time_list, wind_speed,color='b',lw=3)
obs_meteo['Wind Speed (m/s)'].plot(ax=ax1,linestyle=':',color='r',lw=3)

plt.title('Wind speed (m/s) at Ludingvig')
fig_name=os.path.join(Outfig,'Wind_speed_ludingvig.jpg')
fig1.savefig(fig_name, dpi=300)


#%%
# =============================================================================
# test air_pressure
# =============================================================================
# holland: 156832, out: 171742
# ludington: 171597, out: 229471
HRRR_in= Dataset(os.path.join(HRRR_dir,'P_new_fvcom_hrrr.nc'))
print(HRRR_in.variables.keys())  

Atmos_P=ncfile1.variables['atmos_press'][:] 
#%%
for gauge in ['holland','ludington']:
    #%
    P=Atmos_P[:,int(loc_gauge.loc[gauge,'Index_node'])]
    P_out=Atmos_P[:,int(loc_gauge.loc[gauge,'Index2'])]
    P_center=Atmos_P[:,int(125858)]
    
    fig1,ax1 = plt.subplots(1)
    plt.plot(time_list2, P,linestyle='-',color='b',lw=2)
    plt.plot(time_list2, P_out,linestyle='--',color='g',lw=2)
    plt.plot(time_list2, P_center,linestyle=':',color='k',lw=2)

    plt.title('Air Pressure at '+ gauge)
    ax1.legend(labels=['Bob_Mesh','Bob_Mesh_out','Bob_Mesh_center'],ncol=1)
    ax1.set_ylabel('Air Pressure (pa)')
    
    fig_name=os.path.join(Outfig,'P_'+gauge+'_5_1.jpg')
    fig1.savefig(fig_name, dpi=300)


#%%








