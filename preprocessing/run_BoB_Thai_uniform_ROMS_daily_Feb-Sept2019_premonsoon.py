from parcels import FieldSet, ParticleSet, Variable, JITParticle, ScipyParticle, Field, VectorField, plotTrajectoriesFile, ParcelsRandom
from parcels.tools.converters import Geographic, GeographicPolar
import numpy as np
import math
from math import fabs
from datetime import timedelta
from operator import attrgetter
from all_kernels import AdvectionRK4, DiffusionUniformKh, Windage, StokesDrift, BeachedStatusCheck, BeachedDelete

output_filename="/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Thai_uniform_ROMS_daily_Feb-Sept2019_premonsoon.nc"

input_velocities_filepath = "/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/input/processed/paper_data/"

ufile = input_velocities_filepath+'ocean_velocities_u_ROMS_Feb-Sept2019_daily.nc'
vfile = input_velocities_filepath+'ocean_velocities_v_ROMS_Feb-Sept2019_daily.nc'

filenames = {'U': {'lon': ufile, 'lat': ufile, 'data': ufile},
             'V': {'lon': vfile, 'lat': vfile, 'data': vfile}}

variables = {'U': "u",
             'V': "v"}

dimensions = {'U': {'lon': 'lon_u', 'lat': 'lat_u', 'time': 'ocean_time'},
              'V': {'lon': 'lon_v', 'lat': 'lat_v', 'time': 'ocean_time'}}


fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=True)

def DeleteParticle(particle, fieldset, time):
    #print("Particle [%d] lost !! (%g %g %g %g)" % (particle.id, particle.lon, particle.lat, particle.depth, particle.time))
    particle.delete()

# DIFFUSION 
#ParcelsRandom.seed(123456) # this line means that the 'random' walk will always be the same for debugging purposes - the number means the calculated randomness will always start from the same value. 
kh_zonal = 100  # in m^2/s
kh_meridional = 100  # in m^2/s

# create field of constant Kh_zonal and Kh_meridional
fieldset.add_field(Field('Kh_zonal', kh_zonal, mesh='spherical'))
fieldset.add_field(Field('Kh_meridional', kh_meridional, mesh='spherical'))

lons_thailand = [98.46521645, 98.36450539, 98.36, 98.39453419, 98.36330459, 98.31794299, 98.22089368, 98.20894477, 98.19369555, 98.20629535, 98.15644736, 98.15436042, 98.1833423, 98.21616106, 98.21630916, 98.22131056, 98.23537552, 98.3190308, 98.43291856, 98.47338545, 98.49429311, 98.57024252, 98.63553449, 98.69220480000001, 98.795364, 98.89626653, 98.90725136, 98.96455982]
lats_thailand = [9.89068589, 9.82844698, 9.71245446, 9.590927220000001, 9.44697279, 9.30710011, 9.22696962, 9.08140073, 8.93488921, 8.79925959, 8.66841364, 8.52158722, 8.38221815, 8.247399380000001, 8.10021199, 7.95331855, 7.806688599999999, 7.73013195, 7.81121897, 7.94022193, 8.084598380000001, 8.18219522, 8.18147809, 8.05922502, 7.97709803, 7.92481909, 7.80799353, 7.780947869999999]

# BEACHING
def MakeParticleSet(fieldset): # defining a function
        #class BeachableParticles(JITParticle): # defining a class of particle within this function - an instance of this class will be initialised through calling this function
        class BeachableParticles(ScipyParticle):
                beached = Variable('beached', dtype=np.int32, initial=0) # 'beached' is now an available variable of an oject of the class: 'BeachedParticles'
        return ParticleSet.from_list(fieldset, BeachableParticles, # this returns an instance of the BeachableParticles class
                                     #lon=[44, 45, 59, 68.5, 73, 65, 75, 79, 81, 89, 87, 99],
                                     #lat=[0, 12, 24, 22.5, 15, 12, 0, 7, 15, 20, 14, 5],
                                     lon = lons_thailand,
                                     lat = lats_thailand,
                                     repeatdt=timedelta(days=1)) # 2,016 particles in 14 days

				      
# WINDAGE
windage = 0.01
filenames_wind = {'U_wind': input_velocities_filepath + "windage_u_ROMS_interpolated_Feb-Oct2019_daily.nc",
                 'V_wind': input_velocities_filepath + "windage_v_ROMS_interpolated_Feb-Oct2019_daily.nc"}
variables_wind = {'U_wind': '__xarray_dataarray_variable__',
                 'V_wind': '__xarray_dataarray_variable__'}
dimensions_wind = {'lat': 'lat',
                  'lon': 'lon',
                  'time': 'time'}

fieldset_wind = FieldSet.from_netcdf(filenames_wind, variables_wind, dimensions_wind) # this line creates a new fieldset

fieldset_wind.U_wind.units = GeographicPolar() # this converts input units of m/s to degrees/sec, instead of the values being taken as degrees/sec in the first place.
fieldset_wind.V_wind.units = Geographic()

fieldset_wind.U_wind.set_scaling_factor(windage) # this line sets the scaling factor so only a certain percentage of the wind is applied to the particles
fieldset_wind.V_wind.set_scaling_factor(windage) 

fieldset.add_field(fieldset_wind.U_wind) 
fieldset.add_field(fieldset_wind.V_wind)

vectorField_wind = VectorField('UV_wind',fieldset.U_wind,fieldset.V_wind) # this line is a creating a new field. VectorField is a class and this is calling an instance of it.
fieldset.add_vector_field(vectorField_wind) # this line adds it to the existing fieldset which now has U_wind and V_wind and now UV_wind, callable from fieldset.

# STOKES DRIFT
filenames_stokes = {'U_stokes': input_velocities_filepath + "stokes_u_ROMS_interpolated_Feb-Oct2019_daily_fromcoarse.nc",
                 'V_stokes': input_velocities_filepath + "stokes_v_ROMS_interpolated_Feb-Oct2019_daily_fromcoarse.nc"}
variables_stokes = {'U_stokes': '__xarray_dataarray_variable__',
                 'V_stokes': '__xarray_dataarray_variable__'}
dimensions_stokes = {'lat': 'lat',
                  'lon': 'lon',
                  'time': 'time'}

fieldset_stokes = FieldSet.from_netcdf(filenames_stokes, variables_stokes, dimensions_stokes) 

fieldset_stokes.U_stokes.units = GeographicPolar() 
fieldset_stokes.V_stokes.units = Geographic()

fieldset.add_field(fieldset_stokes.U_stokes) 
fieldset.add_field(fieldset_stokes.V_stokes)

vectorField_stokes = VectorField('UV_stokes',fieldset.U_stokes,fieldset.V_stokes) 
fieldset.add_vector_field(vectorField_stokes) 

pset = MakeParticleSet(fieldset)

## add kernel code here or convert functions written in other scripts to a kernel here.
k_DeleteParticle = pset.Kernel(DeleteParticle)
k_AdvectionRK4 = pset.Kernel(AdvectionRK4)
k_DiffusionUniformKh = pset.Kernel(DiffusionUniformKh)
k_BeachedStatusCheck = pset.Kernel(BeachedStatusCheck)
k_BeachedDelete = pset.Kernel(BeachedDelete)
k_Windage = pset.Kernel(Windage)
k_StokesDrift = pset.Kernel(StokesDrift)

output_file = pset.ParticleFile(name=output_filename, outputdt=timedelta(hours=24))

pset.execute(k_AdvectionRK4 + k_DiffusionUniformKh + k_Windage + k_StokesDrift + k_BeachedStatusCheck + k_BeachedDelete,
	     runtime=timedelta(days=120), # 4 months with repeated release of particles
	     dt=timedelta(minutes=15),
	     output_file=output_file,
	     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})	     

# now stop the repeated release
pset.repeatdt = None

# now continue running for the remaining 2 weeks
pset.execute(k_AdvectionRK4 + k_DiffusionUniformKh + k_Windage + k_StokesDrift + k_BeachedStatusCheck + k_BeachedDelete,
             runtime=timedelta(days=122), 
             dt=timedelta(minutes=15),
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

output_file.close()

