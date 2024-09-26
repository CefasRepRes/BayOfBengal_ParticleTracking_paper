from parcels import FieldSet, ParticleSet, Variable, JITParticle, ScipyParticle, Field, VectorField, plotTrajectoriesFile, ParcelsRandom
from parcels.tools.converters import Geographic, GeographicPolar
import numpy as np
import math
from math import fabs
from datetime import timedelta
from operator import attrgetter
from all_kernels import AdvectionRK4, DiffusionUniformKh, Windage, StokesDrift, BeachedStatusCheck, BeachedDelete

output_filename="/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_SL_uniform_Cop_daily_July2020.nc"

input_velocities_filepath = "/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/input/processed/paper_data/"

filenames = {'U': input_velocities_filepath + "ocean_velocities_July2020_daily.nc",
             'V': input_velocities_filepath + "ocean_velocities_July2020_daily.nc"}

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

lons_sl = [81.34851444999998, 81.27342240999998, 81.19681289, 81.11207956, 81.02430821999998, 80.94104212, 80.86771129, 80.77003827, 80.66385470999998, 80.55323756999998, 80.43744234, 80.34537127999998, 80.24715533, 80.10135703, 79.95562433, 79.81326726, 79.79680043, 79.89428526999998, 80.01376150999998, 80.08461515, 79.99123371999998, 80.04902219, 79.99758554, 79.93573334, 79.87918012999998, 79.88502939999998, 79.79575701, 79.75665067, 79.67580469, 79.83, 81.84290754, 81.80636649, 81.72487236, 81.62951657, 81.56877746, 81.49451008, 81.45290325, 81.41878636, 80.09073067, 80.20267271999998, 80.34168651, 80.48401291, 80.63120030999998, 80.76627555, 80.89836107, 81.03826723999998, 81.17849916999998, 81.3084596, 81.44112964999998, 81.55191166, 81.67455796, 81.75909516999998, 81.82888955, 81.87773031, 81.91382092, 81.90807633999998, 81.91172559, 81.89638746999998, 79.64414424, 79.68044222, 79.71170377, 79.74061253, 79.74944734999998, 79.76212694, 79.79230579, 79.79765269, 79.79758957, 79.83239849999998, 79.87699784999998, 79.92143402, 79.96141503, 80.01203999, 79.62, 81.89] 
lats_sl = [8.620542429999999, 8.690163239999999, 8.81545106, 8.93395572, 9.04990911, 9.17088735, 9.297006119999999, 9.4049847, 9.506967909999998, 9.60365539, 9.69470279, 9.80868784, 9.912362759999999, 9.89160984, 9.87015892, 9.839782099999999, 9.695988, 9.549999999999999, 9.59, 9.51081074, 9.40814082, 9.27474618, 9.299999999999999, 9.08, 8.83544858, 8.68826178, 8.57647969, 8.437231299999999, 8.31968274, 8.91, 7.60108421, 7.742429710000001, 7.86491881, 7.973443420000001, 8.10207934, 8.224342109999998, 8.36544195, 8.508739559999999, 6.050999539999999, 5.96020286, 5.91148602, 5.87963259, 5.8737986499999995, 5.92959408, 5.99450919, 6.03748444, 6.07317224, 6.12735316, 6.1822308, 6.2783584, 6.356349209999999, 6.46773445, 6.595394489999999, 6.73325325, 6.875967709999999, 7.02315718, 7.17034422, 7.3140867499999995, 7.98364466, 7.84090359, 7.69695609, 7.55374099, 7.4067032, 7.26016887, 7.1159905, 6.96935191, 6.8221645, 6.6802808, 6.53989574, 6.4010807, 6.25954906, 6.12188802, 8.15, 7.44]

# BEACHING
def MakeParticleSet(fieldset): # defining a function
	#class BeachableParticles(JITParticle): # defining a class of particle within this function - an instance of this class will be initialised through calling this function
	class BeachableParticles(ScipyParticle): 
		beached = Variable('beached', dtype=np.int32, initial=0) # 'beached' is now an available variable of an oject of the class: 'BeachedParticles'
	return ParticleSet.from_list(fieldset, BeachableParticles, # this returns an instance of the BeachableParticles class
                                     #lon=[44, 45, 59, 68.5, 73, 65, 75, 79, 81, 89, 87, 99],
                                     #lat=[0, 12, 24, 22.5, 15, 12, 0, 7, 15, 20, 14, 5],
                                     lon= lons_sl,
                                     lat= lats_sl,
                                     repeatdt=timedelta(hours=1)) # 24,864 particles in 14 days
				      
# WINDAGE
windage = 0.01
filenames_wind = {'U_wind': input_velocities_filepath + "windage_u_avgd_interpolated_July2020_daily.nc",
                 'V_wind': input_velocities_filepath + "windage_v_avgd_interpolated_July2020_daily.nc"}
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
filenames_stokes = {'U_stokes': input_velocities_filepath + "stokes_u_avgd_interpolated_July2020_daily_fromcoarse.nc",
                 'V_stokes': input_velocities_filepath + "stokes_v_avgd_interpolated_July2020_daily_fromcoarse.nc"}
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

output_file = pset.ParticleFile(name=output_filename, outputdt=timedelta(hours=1))

pset.execute(k_AdvectionRK4 + k_DiffusionUniformKh + k_Windage + k_StokesDrift + k_BeachedStatusCheck + k_BeachedDelete,
	     runtime=timedelta(days=14), # 2 weeks with repeated release of particles
	     dt=timedelta(minutes=15),
	     output_file=output_file,
	     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})	     

# now stop the repeated release
pset.repeatdt = None

# now continue running for the remaining 2 weeks
pset.execute(k_AdvectionRK4 + k_DiffusionUniformKh + k_Windage + k_StokesDrift + k_BeachedStatusCheck + k_BeachedDelete,
             runtime=timedelta(days=17), 
             dt=timedelta(minutes=15),
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

output_file.close()


