import time
t1=time.time()

import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import xarray as xr
from netCDF4 import Dataset
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

elapsed1 = time.time() - t1
t2=time.time()
print(f'elapsed1 is {elapsed1}')

# INPUTS
# input files
particle_file_SL1 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_SL_uniform_Cop_daily_Jun2018-Sept2019_monsoon.nc'
particle_file_India1 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_India_uniform_Cop_daily_Jun2018-Sept2019_monsoon.nc'
particle_file_Bang1 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Bang_uniform_Cop_daily_Jun2018-Sept2019_monsoon.nc'
particle_file_Myan1 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Myan_uniform_Cop_daily_Jun2018-Sept2019_monsoon.nc'
particle_file_Thai1 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Thai_uniform_Cop_daily_Jun2018-Sept2019_monsoon.nc'
particle_file_Indo1 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Indonesia_uniform_Cop_daily_Jun2018-Sept2019_monsoon.nc'
velocity_file = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/input/processed/paper_data/ocean_velocities_Cop_Jun2018-Sept2019_daily.nc'

particle_file_SL2 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_SL_uniform_Cop_daily_Oct2018-Sept2019_postmonsoon.nc'
particle_file_India2 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_India_uniform_Cop_daily_Oct2018-Sept2019_postmonsoon.nc'
particle_file_Bang2 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Bang_uniform_Cop_daily_Oct2018-Sept2019_postmonsoon.nc'
particle_file_Myan2 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Myan_uniform_Cop_daily_Oct2018-Sept2019_postmonsoon.nc'
particle_file_Thai2 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Thai_uniform_Cop_daily_Oct2018-Sept2019_postmonsoon.nc'
particle_file_Indo2 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Indonesia_uniform_Cop_daily_Oct2018-Sept2019_postmonsoon.nc'
#velocity_file = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/input/processed/paper_data/ocean_velocities_Cop_Oct2018-Sept2019_daily.nc'

particle_file_SL3 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_SL_uniform_Cop_daily_Feb-Sept2019_premonsoon.nc'
particle_file_India3 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_India_uniform_Cop_daily_Feb-Sept2019_premonsoon.nc'
particle_file_Bang3 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Bang_uniform_Cop_daily_Feb-Sept2019_premonsoon.nc'
particle_file_Myan3 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Myan_uniform_Cop_daily_Feb-Sept2019_premonsoon.nc'
particle_file_Thai3 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Thai_uniform_Cop_daily_Feb-Sept2019_premonsoon.nc'
particle_file_Indo3 = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Indonesia_uniform_Cop_daily_Feb-Sept2019_premonsoon.nc'
#velocity_file = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/input/processed/paper_data/ocean_velocities_Cop_Feb-Sept2019_daily.nc'

# output files - what to call the animation?
fig_name = '/gpfs/home/rpe16nbu/projects/ocpp_mo1/plots/paper_data/BoB_all_countries_uniform_Cop_daily_Jun2018-Sept2019_full year.gif'
# set parameters
interval = 300 # larger number makes video slower
# particle timestep - were they output hourly or daily? Comment or uncomment accordingly. Script assumes ocean velocities are daily.
outputdt = timedelta(days=1) # particle positions output once an day
ntimesteps = 487

## particle info
data_p_SL1 = xr.open_dataset(particle_file_SL1)
lon_p_SL1 = data_p_SL1['lon'].values[:,:]
lat_p_SL1 = data_p_SL1['lat'].values[:,:]
time_p_SL1 = data_p_SL1['time'].values[:,:]
data_p_India1 = xr.open_dataset(particle_file_India1)
lon_p_India1 = data_p_India1['lon'].values[:,:]
lat_p_India1 = data_p_India1['lat'].values[:,:]
time_p_India1 = data_p_India1['time'].values[:,:]
data_p_Bang1 = xr.open_dataset(particle_file_Bang1)
lon_p_Bang1 = data_p_Bang1['lon'].values[:,:]
lat_p_Bang1 = data_p_Bang1['lat'].values[:,:]
time_p_Bang1 = data_p_Bang1['time'].values[:,:]
data_p_Myan1 = xr.open_dataset(particle_file_Myan1)
lon_p_Myan1 = data_p_Myan1['lon'].values[:,:]
lat_p_Myan1 = data_p_Myan1['lat'].values[:,:]
time_p_Myan1 = data_p_Myan1['time'].values[:,:]
data_p_Thai1 = xr.open_dataset(particle_file_Thai1)
lon_p_Thai1 = data_p_Thai1['lon'].values[:,:]
lat_p_Thai1 = data_p_Thai1['lat'].values[:,:]
time_p_Thai1 = data_p_Thai1['time'].values[:,:]
data_p_Indo1 = xr.open_dataset(particle_file_Indo1)
lon_p_Indo1 = data_p_Indo1['lon'].values[:,:]
lat_p_Indo1 = data_p_Indo1['lat'].values[:,:]
time_p_Indo1 = data_p_Indo1['time'].values[:,:]

data_p_SL2 = xr.open_dataset(particle_file_SL2)
lon_p_SL2 = data_p_SL2['lon'].values[:,:]
lat_p_SL2 = data_p_SL2['lat'].values[:,:]
time_p_SL2 = data_p_SL2['time'].values[:,:]
data_p_India2 = xr.open_dataset(particle_file_India2)
lon_p_India2 = data_p_India2['lon'].values[:,:]
lat_p_India2 = data_p_India2['lat'].values[:,:]
time_p_India2 = data_p_India2['time'].values[:,:]
data_p_Bang2 = xr.open_dataset(particle_file_Bang2)
lon_p_Bang2 = data_p_Bang2['lon'].values[:,:]
lat_p_Bang2 = data_p_Bang2['lat'].values[:,:]
time_p_Bang2 = data_p_Bang2['time'].values[:,:]
data_p_Myan2 = xr.open_dataset(particle_file_Myan2)
lon_p_Myan2 = data_p_Myan2['lon'].values[:,:]
lat_p_Myan2 = data_p_Myan2['lat'].values[:,:]
time_p_Myan2 = data_p_Myan2['time'].values[:,:]
data_p_Thai2 = xr.open_dataset(particle_file_Thai2)
lon_p_Thai2 = data_p_Thai2['lon'].values[:,:]
lat_p_Thai2 = data_p_Thai2['lat'].values[:,:]
time_p_Thai2 = data_p_Thai2['time'].values[:,:]
data_p_Indo2 = xr.open_dataset(particle_file_Indo2)
lon_p_Indo2 = data_p_Indo2['lon'].values[:,:]
lat_p_Indo2 = data_p_Indo2['lat'].values[:,:]
time_p_Indo2 = data_p_Indo2['time'].values[:,:]

data_p_SL3 = xr.open_dataset(particle_file_SL3)
lon_p_SL3 = data_p_SL3['lon'].values[:,:]
lat_p_SL3 = data_p_SL3['lat'].values[:,:]
time_p_SL3 = data_p_SL3['time'].values[:,:]
data_p_India3 = xr.open_dataset(particle_file_India3)
lon_p_India3 = data_p_India3['lon'].values[:,:]
lat_p_India3 = data_p_India3['lat'].values[:,:]
time_p_India3 = data_p_India3['time'].values[:,:]
data_p_Bang3 = xr.open_dataset(particle_file_Bang3)
lon_p_Bang3 = data_p_Bang3['lon'].values[:,:]
lat_p_Bang3 = data_p_Bang3['lat'].values[:,:]
time_p_Bang3 = data_p_Bang3['time'].values[:,:]
data_p_Myan3 = xr.open_dataset(particle_file_Myan3)
lon_p_Myan3 = data_p_Myan3['lon'].values[:,:]
lat_p_Myan3 = data_p_Myan3['lat'].values[:,:]
time_p_Myan3 = data_p_Myan3['time'].values[:,:]
data_p_Thai3 = xr.open_dataset(particle_file_Thai3)
lon_p_Thai3 = data_p_Thai3['lon'].values[:,:]
lat_p_Thai3 = data_p_Thai3['lat'].values[:,:]
time_p_Thai3 = data_p_Thai3['time'].values[:,:]
data_p_Indo3 = xr.open_dataset(particle_file_Indo3)
lon_p_Indo3 = data_p_Indo3['lon'].values[:,:]
lat_p_Indo3 = data_p_Indo3['lat'].values[:,:]
time_p_Indo3 = data_p_Indo3['time'].values[:,:]

# np.set_printoptions(threshold=sys.maxsize)
# print(data_p['beached'].data) # .values and .data give same result
#time_max = max(len(time_p_SL[1]),len(time_p_India[1]),len(time_p_Bang[1]),len(time_p_Myan[1]),len(time_p_Thai[1]),len(time_p_Indo[1]))

#monsoon
timerange1 = np.arange(np.nanmin(data_p_India1['time'].values), 				# start 
                      np.nanmax(data_p_India1['time'].values)+np.timedelta64(outputdt),   	# stop
                      outputdt)  # timerange in nanoseconds 				# step
#postmonsoon
timerange2 = np.arange(np.nanmin(data_p_Myan2['time'].values),                              # start
                      np.nanmax(data_p_Myan3['time'].values)+np.timedelta64(outputdt),     # stop
                      outputdt)  # timerange in nanoseconds                             # step
#premonsoon
#timerange3 = np.arange(np.nanmin(data_p_Thai3['time'].values),                              # start
#                      np.nanmax(data_p_Thai3['time'].values)+np.timedelta64(outputdt),     # stop
#                      outputdt)  # timerange in nanoseconds                             # step
timerange3 = np.arange(np.nanmin(data_p_SL3['time'].values),                              # start
                      np.nanmax(data_p_SL3['time'].values)+np.timedelta64(outputdt),     # stop
                      outputdt)  # timerange in nanoseconds                             # step

#print(timerange)


time_id_SL1 = np.where(data_p_SL1['time'] == timerange1[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_India1 = np.where(data_p_India1['time'] == timerange1[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Bang1 = np.where(data_p_Bang1['time'] == timerange1[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Myan1 = np.where(data_p_Myan1['time'] == timerange1[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Thai1 = np.where(data_p_Thai1['time'] == timerange1[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Indo1 = np.where(data_p_Indo1['time'] == timerange1[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.

time_id_SL2 = np.where(data_p_SL2['time'] == timerange2[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_India2 = np.where(data_p_India2['time'] == timerange2[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Bang2 = np.where(data_p_Bang2['time'] == timerange2[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Myan2 = np.where(data_p_Myan2['time'] == timerange2[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Thai2 = np.where(data_p_Thai2['time'] == timerange2[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Indo2 = np.where(data_p_Indo2['time'] == timerange2[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.

time_id_SL3 = np.where(data_p_SL3['time'] == timerange3[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_India3 = np.where(data_p_India3['time'] == timerange3[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Bang3 = np.where(data_p_Bang3['time'] == timerange3[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Myan3 = np.where(data_p_Myan3['time'] == timerange3[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Thai3 = np.where(data_p_Thai3['time'] == timerange3[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.
time_id_Indo3 = np.where(data_p_Indo3['time'] == timerange3[0]) # Indices of the data where time = 0, so can use this to index all particles in data_p that were released at time0, used to create the first frame.



elapsed2 = time.time() - t2
t3=time.time()
print(f'elapsed2 is {elapsed2}')
#print(f'np.shape(lon_p_Thai) is {np.shape(lon_p_Thai)}')
#print(f'np.shape(lat_p_Thai) is {np.shape(lat_p_Thai)}')
#print(f'np.shape(time_p_Thai) is {np.shape(time_p_Thai)}')
#print(f'lon_p_Thai is lon_p_Thai}')
#print(f'lat_p_Thai is lat_p_Thai}')
#print(f'time_p_Thai is time_p_Thai}')

# ocean velocities info
openfile_u = Dataset(velocity_file)
lon_vec_u = openfile_u.variables['longitude']
lat_vec_u = openfile_u.variables['latitude']
time_u = openfile_u.variables['time']
u =  openfile_u.variables['uo']
v =  openfile_u.variables['vo']

elapsed3 = time.time() - t3
t4=time.time()
print(f'elapsed3 is {elapsed3}')

# create grids of coordinates rather than just lists
lon_grid_u, lat_grid_u = np.meshgrid(lon_vec_u, lat_vec_u)

# defining datetime objects for start and end time of simulation
interval_type = 'hours'
interval_num_start = int(time_u[0])
interval_num_end = int(time_u[-1])

origin_time_u = datetime.datetime.strptime('1950/01/01 00:00:00','%Y/%m/%d %H:%M:%S') # simulation time output is number of hours since 00:00:00 01-01-1950
start_time_u = origin_time_u + datetime.timedelta(**{interval_type: interval_num_start}) # turns integers into a datetime object to be added to origin_time
end_time_u = origin_time_u + datetime.timedelta(**{interval_type: interval_num_end})
times_daily = []
#times_daily.append(start_time_u)
for day in range(len(time_u)):
        interval_num_day = int(time_u[day])
        times_daily.append(origin_time_u + datetime.timedelta(**{interval_type: int(time_u[day])}))

elapsed4 = time.time() - t4
t5=time.time()
print(f'elapsed4 is {elapsed4}')

# u and p are different output freqs - to get def animate(ii) to loop over same number of iis, need to reduce/subset u
if outputdt == timedelta(days=24):
    u_subset = u[0:-1:24,0,:-1,:-1] # setting the subset to [0:-1:24, 0, :, :] (i.e. without the minus ones in the last two dimensions messes up the projection)
    v_subset = v[0:-1:24,0,:-1,:-1] # setting the subset to [0:-1:24, 0, :, :] (i.e. without the minus ones in the last two dimensions messes up the projection)
    time_subset = times_hourly[0:-1] # if animation is going to show daily rather than hourly frames
elif outputdt == timedelta(days=1):
    u_subset = u[0:-1,0,:-1,:-1]
    v_subset = v[0:-1,0,:-1,:-1]
    #time_subset = times_hourly[0:-1:24] # if animation is going to show daily rather than hourly frames
    time_subset = times_daily[0:-1] # if animation is going to show hourly frames

speed_subset = np.sqrt(np.abs(u_subset)**2 + np.abs(v_subset)**2)

if ntimesteps:
    speed_subset = speed_subset[:ntimesteps,:,:]
    time_subset = time_subset[:ntimesteps]

#print(np.shape(speed_subset))
#print(np.shape(time_subset))
#print(np.shape(timerange))
#print(np.shape(time_id))

# create animation
# set up figure and axis
fig, ax = plt.subplots(figsize=(20, 15), dpi=300)

# mappable object for u velocity, first frame
m = ax.pcolormesh(lon_grid_u, lat_grid_u, speed_subset[0,:,:], vmin=0, vmax=np.amax(speed_subset), cmap='Blues_r') # colorbar needs changing to one that doesn't go white at end
cbar = fig.colorbar(m)
cbar.set_label('m/s', fontsize=25)#, fontweight='bold')
cbar.ax.tick_params(labelsize=20)

#title = ax.text(1,1, "test title", fontsize='x-large', c='red')#, bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="center")
title = ax.set_title('Position at ' + str(start_time_u), fontsize=25, fontweight='bold')
fig.text(0.4, 0.94, 'CMEMS', fontsize=30, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=25)#, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=25)#, fontweight='bold')
ax.set_yticklabels(ax.get_yticks(), fontsize=20)#, fontweight='bold')
ax.set_xticklabels(ax.get_xticks(), fontsize=20)#, fontweight='bold')

# scatter plot, first frame
# separate colours don't work because they change when other particles get beacehed and disappear. ANd different colours dont start from the same region because of this.
scat_SL1 = ax.scatter(lon_p_SL1[time_id_SL1], lat_p_SL1[time_id_SL1], color='r', s=10) # plot particle 1 start point
scat_India1 = ax.scatter(lon_p_India1[time_id_India1], lat_p_India1[time_id_India1],color='yellow', s=10)
scat_Bang1 = ax.scatter(lon_p_Bang1[time_id_Bang1], lat_p_Bang1[time_id_Bang1],color='pink', s=10)
scat_Myan1 = ax.scatter(lon_p_Myan1[time_id_Myan1], lat_p_Myan1[time_id_Myan1],color='lime', s=10)
scat_Thai1 = ax.scatter(lon_p_Thai1[time_id_Thai1], lat_p_Thai1[time_id_Thai1],color='grey', s=10)
scat_Indo1 = ax.scatter(lon_p_Indo1[time_id_Indo1], lat_p_Indo1[time_id_Indo1],color='darkorange', s=10)
scat_SL2 = ax.scatter(lon_p_SL2[time_id_SL2], lat_p_SL2[time_id_SL2], color='r', s=10) # plot particle 1 start point
scat_India2 = ax.scatter(lon_p_India2[time_id_India2], lat_p_India2[time_id_India2],color='yellow', s=10)
scat_Bang2 = ax.scatter(lon_p_Bang2[time_id_Bang2], lat_p_Bang2[time_id_Bang2],color='pink', s=10)
scat_Myan2 = ax.scatter(lon_p_Myan2[time_id_Myan2], lat_p_Myan2[time_id_Myan2],color='lime', s=10)
scat_Thai2 = ax.scatter(lon_p_Thai2[time_id_Thai2], lat_p_Thai2[time_id_Thai2],color='grey', s=10)
scat_Indo2 = ax.scatter(lon_p_Indo2[time_id_Indo2], lat_p_Indo2[time_id_Indo2],color='darkorange', s=10)
scat_SL3 = ax.scatter(lon_p_SL3[time_id_SL3], lat_p_SL3[time_id_SL3], color='r', s=10) # plot particle 1 start point
scat_India3 = ax.scatter(lon_p_India3[time_id_India3], lat_p_India3[time_id_India3],color='yellow', s=10)
scat_Bang3 = ax.scatter(lon_p_Bang3[time_id_Bang3], lat_p_Bang3[time_id_Bang3],color='pink', s=10)
scat_Myan3 = ax.scatter(lon_p_Myan3[time_id_Myan3], lat_p_Myan3[time_id_Myan3],color='lime', s=10)
scat_Thai3 = ax.scatter(lon_p_Thai3[time_id_Thai3], lat_p_Thai3[time_id_Thai3],color='grey', s=10)
scat_Indo3 = ax.scatter(lon_p_Indo3[time_id_Indo3], lat_p_Indo3[time_id_Indo3],color='darkorange', s=10)

# Create custom legend handles
legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='Sri Lanka', markerfacecolor=scat_SL1.get_facecolor()[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='India', markerfacecolor=scat_India1.get_facecolor()[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Bangladesh', markerfacecolor=scat_Bang1.get_facecolor()[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Myanmar', markerfacecolor=scat_Myan1.get_facecolor()[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Thailand', markerfacecolor=scat_Thai1.get_facecolor()[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Indonesia', markerfacecolor=scat_Indo1.get_facecolor()[0], markersize=10)
]

# Add legend to the plot with the custom handles
plt.legend(handles=legend_handles,
           scatterpoints=1,
           loc='upper left',
           ncol=1,
           fontsize=20)

elapsed5 = time.time() - t5
t6=time.time()
print(f'elapsed5 is {elapsed5}')

#Hourly
# create extra frames for rest of run
def animate(ii):
    print(ii)
    m.set_array(speed_subset[ii, :, :]) # Not sure exactly what set_array does but it somehow updates the m (mappable) values to the next frame (i.e. 24 hours later in this case)
    time_id_SL1 = np.where(time_p_SL1 == timerange1[ii])
    time_id_India1 = np.where(time_p_India1 == timerange1[ii])
    time_id_Bang1 = np.where(time_p_Bang1 == timerange1[ii])
    time_id_Myan1 = np.where(time_p_Myan1 == timerange1[ii])
    time_id_Thai1 = np.where(time_p_Thai1 == timerange1[ii])
    time_id_Indo1 = np.where(time_p_Indo1 == timerange1[ii])
    if ii>122:
        time_id_SL2 = np.where(time_p_SL2 == timerange2[ii-122])
        time_id_India2 = np.where(time_p_India2 == timerange2[ii-122])
        time_id_Bang2 = np.where(time_p_Bang2 == timerange2[ii-122])
        time_id_Myan2 = np.where(time_p_Myan2 == timerange2[ii-122])
        time_id_Thai2 = np.where(time_p_Thai2 == timerange2[ii-122])
        time_id_Indo2 = np.where(time_p_Indo2 == timerange2[ii-122])
    if ii>245:
        time_id_SL3 = np.where(time_p_SL3 == timerange3[ii-245])
        time_id_India3 = np.where(time_p_India3 == timerange3[ii-245])
        time_id_Bang3 = np.where(time_p_Bang3 == timerange3[ii-245])
        time_id_Myan3 = np.where(time_p_Myan3 == timerange3[ii-245])
        time_id_Thai3 = np.where(time_p_Thai3 == timerange3[ii-245])
        time_id_Indo3 = np.where(time_p_Indo3 == timerange3[ii-245])

    # Must pass scat.set_offsets an N x 2 array
    scat_SL1.set_offsets(np.c_[lon_p_SL1[time_id_SL1], lat_p_SL1[time_id_SL1]]) # updates particles' position
    scat_India1.set_offsets(np.c_[lon_p_India1[time_id_India1], lat_p_India1[time_id_India1]]) # updates particles' position
    scat_Bang1.set_offsets(np.c_[lon_p_Bang1[time_id_Bang1], lat_p_Bang1[time_id_Bang1]]) # updates particles' position
    scat_Myan1.set_offsets(np.c_[lon_p_Myan1[time_id_Myan1], lat_p_Myan1[time_id_Myan1]]) # updates particles' position
    scat_Thai1.set_offsets(np.c_[lon_p_Thai1[time_id_Thai1], lat_p_Thai1[time_id_Thai1]]) # updates particles' position
    scat_Indo1.set_offsets(np.c_[lon_p_Indo1[time_id_Indo1], lat_p_Indo1[time_id_Indo1]]) # updates particles' position
    if ii>122:
        scat_SL2.set_offsets(np.c_[lon_p_SL2[time_id_SL2], lat_p_SL2[time_id_SL2]]) # updates particles' position
        scat_India2.set_offsets(np.c_[lon_p_India2[time_id_India2], lat_p_India2[time_id_India2]]) # updates particles' position
        scat_Bang2.set_offsets(np.c_[lon_p_Bang2[time_id_Bang2], lat_p_Bang2[time_id_Bang2]]) # updates particles' position
        scat_Myan2.set_offsets(np.c_[lon_p_Myan2[time_id_Myan2], lat_p_Myan2[time_id_Myan2]]) # updates particles' position
        scat_Thai2.set_offsets(np.c_[lon_p_Thai2[time_id_Thai2], lat_p_Thai2[time_id_Thai2]]) # updates particles' position
        scat_Indo2.set_offsets(np.c_[lon_p_Indo2[time_id_Indo2], lat_p_Indo2[time_id_Indo2]]) # updates particles' position
    if ii>245:
        scat_SL3.set_offsets(np.c_[lon_p_SL3[time_id_SL3], lat_p_SL3[time_id_SL3]]) # updates particles' position
        scat_India3.set_offsets(np.c_[lon_p_India3[time_id_India3], lat_p_India3[time_id_India3]]) # updates particles' position
        scat_Bang3.set_offsets(np.c_[lon_p_Bang3[time_id_Bang3], lat_p_Bang3[time_id_Bang3]]) # updates particles' position
        scat_Myan3.set_offsets(np.c_[lon_p_Myan3[time_id_Myan3], lat_p_Myan3[time_id_Myan3]]) # updates particles' position
        scat_Thai3.set_offsets(np.c_[lon_p_Thai3[time_id_Thai3], lat_p_Thai3[time_id_Thai3]]) # updates particles' position
        scat_Indo3.set_offsets(np.c_[lon_p_Indo3[time_id_Indo3], lat_p_Indo3[time_id_Indo3]]) # updates particles' position
    title = ax.set_title('Position at ' + str(time_subset[ii]), fontsize=25)

# Save the animation
anim = FuncAnimation(fig, animate, interval=interval, frames=range(0, len(speed_subset)-1, 7), repeat=True) # interval is number of milliseconds between frames
#anim = FuncAnimation(fig, animate, interval=interval, frames=len(speed_subset)-1, repeat=True) # interval is number of milliseconds between frames
anim.save(fig_name, writer='imagemagick')

elapsed6 = time.time() - t6
t7=time.time()
print(f'elapsed6 is {elapsed6}')

# still
fig, ax = plt.subplots(figsize=(20, 15), dpi=300)

# mappable object for u velocity, first frame
m = ax.pcolormesh(lon_grid_u, lat_grid_u, speed_subset[0,:,:], vmin=0, vmax=np.amax(speed_subset), cmap='Blues_r') # colorbar needs changing to one that doesn't go white at end
cbar = fig.colorbar(m)
cbar.set_label('m/s', fontsize=25)#, fontweight='bold')
cbar.ax.tick_params(labelsize=20)

#title = ax.text(1,1, "test title", fontsize='x-large', c='red')#, bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="center")
title = ax.set_title('Position at ' + str(time_subset[200]), fontsize=25)
fig.text(0.4, 0.94, 'CMEMS', fontsize=30, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=25)#, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=25)#, fontweight='bold')
ax.set_yticklabels(ax.get_yticks(), fontsize=20)#, fontweight='bold')
ax.set_xticklabels(ax.get_xticks(), fontsize=20)#, fontweight='bold')

time_id_SL1 = np.where(time_p_SL1 == timerange1[200])
time_id_India1 = np.where(time_p_India1 == timerange1[200])
time_id_Bang1 = np.where(time_p_Bang1 == timerange1[200])
time_id_Myan1 = np.where(time_p_Myan1 == timerange1[200])
time_id_Thai1 = np.where(time_p_Thai1 == timerange1[200])
time_id_Indo1 = np.where(time_p_Indo1 == timerange1[200])
time_id_SL2 = np.where(time_p_SL2 == timerange2[200-122])
time_id_India2 = np.where(time_p_India2 == timerange2[200-122])
time_id_Bang2 = np.where(time_p_Bang2 == timerange2[200-122])
time_id_Myan2 = np.where(time_p_Myan2 == timerange2[200-122])
time_id_Thai2 = np.where(time_p_Thai2 == timerange2[200-122])
time_id_Indo2 = np.where(time_p_Indo2 == timerange2[200-122])

# scatter plot, first frame
# separate colours don't work because they change when other particles get beacehed and disappear. ANd different colours dont start from the same region because of this.
scat_SL1 = ax.scatter(lon_p_SL1[time_id_SL1], lat_p_SL1[time_id_SL1], color='r', s=10) # plot particle 1 start point
scat_India1 = ax.scatter(lon_p_India1[time_id_India1], lat_p_India1[time_id_India1],color='yellow', s=10)
scat_Bang1 = ax.scatter(lon_p_Bang1[time_id_Bang1], lat_p_Bang1[time_id_Bang1],color='pink', s=10)
scat_Myan1 = ax.scatter(lon_p_Myan1[time_id_Myan1], lat_p_Myan1[time_id_Myan1],color='lime', s=10)
scat_Thai1 = ax.scatter(lon_p_Thai1[time_id_Thai1], lat_p_Thai1[time_id_Thai1],color='grey', s=10)
scat_Indo1 = ax.scatter(lon_p_Indo1[time_id_Indo1], lat_p_Indo1[time_id_Indo1],color='darkorange', s=10)
scat_SL2 = ax.scatter(lon_p_SL2[time_id_SL2], lat_p_SL2[time_id_SL2], color='r', s=10) # plot particle 1 start point
scat_India2 = ax.scatter(lon_p_India2[time_id_India2], lat_p_India2[time_id_India2],color='yellow', s=10)
scat_Bang2 = ax.scatter(lon_p_Bang2[time_id_Bang2], lat_p_Bang2[time_id_Bang2],color='pink', s=10)
scat_Myan2 = ax.scatter(lon_p_Myan2[time_id_Myan2], lat_p_Myan2[time_id_Myan2],color='lime', s=10)
scat_Thai2 = ax.scatter(lon_p_Thai2[time_id_Thai2], lat_p_Thai2[time_id_Thai2],color='grey', s=10)
scat_Indo2 = ax.scatter(lon_p_Indo2[time_id_Indo2], lat_p_Indo2[time_id_Indo2],color='darkorange', s=10)

# Create custom legend handles
legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='Sri Lanka', markerfacecolor=scat_SL1.get_facecolor()[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='India', markerfacecolor=scat_India1.get_facecolor()[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Bangladesh', markerfacecolor=scat_Bang1.get_facecolor()[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Myanmar', markerfacecolor=scat_Myan1.get_facecolor()[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Thailand', markerfacecolor=scat_Thai1.get_facecolor()[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Indonesia', markerfacecolor=scat_Indo1.get_facecolor()[0], markersize=10)
]

# Add legend to the plot with the custom handles
plt.legend(handles=legend_handles,
           scatterpoints=1,
           loc='upper left',
           ncol=1,
           fontsize=20)

fig.savefig("/gpfs/home/rpe16nbu/projects/ocpp_mo1/plots/paper_data/CMEMS_anim_still", bbox_inches='tight', facecolor='white', transparent=False)

