from parcels import FieldSet, ParticleSet, Variable, JITParticle, ScipyParticle, Field, VectorField, plotTrajectoriesFile, ParcelsRandom
from parcels.tools.converters import Geographic, GeographicPolar
import numpy as np
import math
from math import fabs
from datetime import timedelta
from operator import attrgetter
from all_kernels import AdvectionRK4, DiffusionUniformKh, Windage, StokesDrift, BeachedStatusCheck, BeachedDelete

output_filename="/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Bang_uniform_ROMS_daily_Jun2018-Sept2019_monsoon.nc"

input_velocities_filepath = "/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/input/processed/paper_data/"

ufile = input_velocities_filepath+'ocean_velocities_u_ROMS_Jun2018-Dec2019_daily.nc'
vfile = input_velocities_filepath+'ocean_velocities_v_ROMS_Jun2018-Dec2019_daily.nc'

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

lons_bangladesh = [89.26927954999998, 89.35697772, 89.49169817, 89.63707816999998, 89.7438559, 89.87625038, 89.97709027, 90.07614718, 90.2206826, 90.36126593, 90.4775672, 90.59840999, 90.73, 90.80604916, 90.92812713, 91.05858738, 91.16875691000001, 91.22998618, 91.24282517, 91.28158618, 91.30915983000001, 91.3424948, 91.42593477, 91.56187851, 91.62957716, 91.67197748, 91.7097221, 91.75624203, 91.7826884, 91.77689761, 91.77111973, 91.80338334, 91.82661580000001, 91.93407605, 91.98254474, 92.03265574, 92.11476927, 92.18139551, 92.25285977, 92.36395711]
lats_bangladesh = [21.574118730000002, 21.67609405, 21.66682335, 21.66263511, 21.75179354, 21.795917510000002, 21.85224401, 21.75148932, 21.747948240000003, 21.78744086, 21.796886750000002, 21.867337600000003, 21.95756528, 22.0353029, 22.020000000000003, 22.04607223, 22.10234595, 22.228418950000002, 22.37516133, 22.50967904, 22.58411749, 22.474545680000002, 22.353156780000003, 22.327339180000003, 22.437340250000002, 22.420524320000002, 22.285531170000002, 22.149603300000003, 22.005564640000003, 21.85837554, 21.711185930000003, 21.57042959, 21.426379410000003, 21.33730702, 21.20162184, 21.07035154, 20.95023966, 20.82135607, 20.69443366, 20.60038093]

# BEACHING
def MakeParticleSet(fieldset): # defining a function
        #class BeachableParticles(JITParticle): # defining a class of particle within this function - an instance of this class will be initialised through calling this function
        class BeachableParticles(ScipyParticle):
                beached = Variable('beached', dtype=np.int32, initial=0) # 'beached' is now an available variable of an oject of the class: 'BeachedParticles'
        return ParticleSet.from_list(fieldset, BeachableParticles, # this returns an instance of the BeachableParticles class
                                     #lon=[44, 45, 59, 68.5, 73, 65, 75, 79, 81, 89, 87, 99],
                                     #lat=[0, 12, 24, 22.5, 15, 12, 0, 7, 15, 20, 14, 5],
                                     lon = lons_bangladesh,
                                     lat = lats_bangladesh,
                                     repeatdt=timedelta(days=1)) # 2,016 particles in 14 days

				      
# WINDAGE
windage = 0.01
filenames_wind = {'U_wind': input_velocities_filepath + "windage_u_ROMS_interpolated_Jun2018-Oct2019_daily.nc",
                 'V_wind': input_velocities_filepath + "windage_v_ROMS_interpolated_Jun2018-Oct2019_daily.nc"}
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
filenames_stokes = {'U_stokes': input_velocities_filepath + "stokes_u_ROMS_interpolated_Jun2018-Oct2019_daily_fromcoarse.nc",
                 'V_stokes': input_velocities_filepath + "stokes_v_ROMS_interpolated_Jun2018-Oct2019_daily_fromcoarse.nc"}
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
	     runtime=timedelta(days=122), # 4 months with repeated release of particles
	     dt=timedelta(minutes=15),
	     output_file=output_file,
	     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})	     

# now stop the repeated release
pset.repeatdt = None

# now continue running for the remaining 2 weeks
pset.execute(k_AdvectionRK4 + k_DiffusionUniformKh + k_Windage + k_StokesDrift + k_BeachedStatusCheck + k_BeachedDelete,
             runtime=timedelta(days=365), 
             dt=timedelta(minutes=15),
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

output_file.close()

