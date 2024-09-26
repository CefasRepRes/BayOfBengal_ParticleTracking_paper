from parcels import FieldSet, ParticleSet, Variable, JITParticle, ScipyParticle, Field, VectorField, plotTrajectoriesFile, ParcelsRandom
from parcels.tools.converters import Geographic, GeographicPolar
import numpy as np
import math
from math import fabs
from datetime import timedelta
from operator import attrgetter
from all_kernels import AdvectionRK4, DiffusionUniformKh, Windage, StokesDrift, BeachedStatusCheck, BeachedDelete

output_filename="/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Indonesia_uniform_Cop_daily_Oct2018-Sept2019_postmonsoon.nc"

input_velocities_filepath = "/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/input/processed/paper_data/"

filenames = {'U': input_velocities_filepath + "ocean_velocities_Cop_Oct2018-Sept2019_daily.nc",
             'V': input_velocities_filepath + "ocean_velocities_Cop_Oct2018-Sept2019_daily.nc"}

variables = {'U': "uo",
             'V': "vo"}

dimensions = {'lat': "latitude",
              'lon': "longitude",
              #'depth': "depth",
              'time': "time"}

chunksize_mb = 256
cs = {'time': ('time', 1), 'lat': ('latitude', chunksize_mb), 'lon': ('longitude', chunksize_mb)}
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, chunksize=cs, allow_time_extrapolation=True)

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

lons_indonesia = [98.94517743, 98.81106944, 98.73556902, 98.62149351, 98.50281264, 98.37698504, 98.30436201, 98.33671613000001, 98.24026587, 98.1140017, 98.03118446, 97.98931653, 97.90415911, 97.78837856, 97.67369556000001, 97.57358352, 97.44152138, 97.29646788, 97.18786234, 97.05654287, 96.91577475, 96.76886032, 96.62656937000001, 96.48138082, 96.33752427, 96.19371585, 96.06475687, 95.9617209, 95.86602219, 95.73529583, 95.58971585, 95.45151268, 95.31978723, 95.17869835, 95.15232225, 95.17315644, 95.20218757, 95.24999061, 95.32007633, 95.38379838, 95.45704523, 95.56708831, 95.68409563, 95.78818678, 95.90414469, 95.99737913000001, 96.12450519, 96.23254833, 96.31416992, 96.41869817000001, 96.53151245, 96.67106002000001, 96.80375617, 96.88556053, 96.83]
lats_indonesia = [3.708721, 3.76624422, 3.88699673, 3.96311196, 4.03944123, 4.11400171, 4.22726002, 4.36256362, 4.51820475, 4.58463084, 4.6769848, 4.81760005, 4.93353515, 5.02460123, 5.11685378, 5.22490814, 5.2493481399999995, 5.22370349, 5.21553837, 5.27674956, 5.29850206, 5.28781024, 5.25452461, 5.24618711, 5.2713788699999995, 5.29734724, 5.36068632, 5.45701466, 5.5591161399999995, 5.62695583, 5.64003931, 5.6742498, 5.62527808, 5.54897181, 5.43151639, 5.29160836, 5.1502856, 5.01106561, 4.88626587, 4.75743127, 4.63289189, 4.53643278, 4.45024449, 4.34616418, 4.25609009, 4.14400586, 4.0722092, 3.97208521, 3.85260865, 3.75314493, 3.66450811, 3.70823762, 3.66437401, 3.54828941, 3.6]

# BEACHING
def MakeParticleSet(fieldset): # defining a function
	#class BeachableParticles(JITParticle): # defining a class of particle within this function - an instance of this class will be initialised through calling this function
	class BeachableParticles(ScipyParticle): 
		beached = Variable('beached', dtype=np.int32, initial=0) # 'beached' is now an available variable of an oject of the class: 'BeachedParticles'
	return ParticleSet.from_list(fieldset, BeachableParticles, # this returns an instance of the BeachableParticles class
                                     #lon=[44, 45, 59, 68.5, 73, 65, 75, 79, 81, 89, 87, 99],
                                     #lat=[0, 12, 24, 22.5, 15, 12, 0, 7, 15, 20, 14, 5],
                                     lon = lons_indonesia,
                                     lat = lats_indonesia,
                                     repeatdt=timedelta(days=1)) # 53,424 particles in 14 days
				      
# WINDAGE
windage = 0.01
filenames_wind = {'U_wind': input_velocities_filepath + "windage_u_avgd_interpolated_Oct2018-Oct2019_daily.nc",
                 'V_wind': input_velocities_filepath + "windage_v_avgd_interpolated_Oct2018-Oct2019_daily.nc"}
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
filenames_stokes = {'U_stokes': input_velocities_filepath + "stokes_u_avgd_interpolated_Oct2018-Oct2019_daily_fromcoarse.nc",
                 'V_stokes': input_velocities_filepath + "stokes_v_avgd_interpolated_Oct2018-Oct2019_daily_fromcoarse.nc"}
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
	     runtime=timedelta(days=123), # June-Sept with repeated release of particles
	     dt=timedelta(minutes=15),
	     output_file=output_file,
	     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})	     

# now stop the repeated release
pset.repeatdt = None

# now continue running for the remaining 2 weeks
pset.execute(k_AdvectionRK4 + k_DiffusionUniformKh + k_Windage + k_StokesDrift + k_BeachedStatusCheck + k_BeachedDelete,
             runtime=timedelta(days=242), 
             dt=timedelta(minutes=15),
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

output_file.close()


