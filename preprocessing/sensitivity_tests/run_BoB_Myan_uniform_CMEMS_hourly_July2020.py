from parcels import FieldSet, ParticleSet, Variable, JITParticle, ScipyParticle, Field, VectorField, plotTrajectoriesFile, ParcelsRandom
from parcels.tools.converters import Geographic, GeographicPolar
import numpy as np
import math
from math import fabs
from datetime import timedelta
from operator import attrgetter
from all_kernels import AdvectionRK4, DiffusionUniformKh, Windage, StokesDrift, BeachedStatusCheck, BeachedDelete

output_filename="/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_Myan_uniform_Cop_hourly_July2020.nc"

input_velocities_filepath = "/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/input/processed/paper_data/"

filenames = {'U': input_velocities_filepath + "ocean_velocities_July2020_hourly.nc",
             'V': input_velocities_filepath + "ocean_velocities_July2020_hourly.nc"}

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

lons_myanmar = [92.45986497, 92.54757118, 92.61289418, 92.70281148, 92.78736917, 92.86924411, 92.93689755, 93.04562247, 93.15993007, 93.29170907, 93.26743309, 93.33058904, 93.35894504, 93.37538354, 93.45268274, 93.43324449, 93.40787607, 93.49512037, 93.57259972000001, 93.65043568, 93.77167338, 93.85994179, 93.93521598, 94.05401398, 94.10836586, 94.19165883, 94.22646538, 94.27245611, 94.31738965, 94.41256499, 94.39161991, 94.42080451, 94.46216735, 94.48554798, 94.48733938, 94.45334140000001, 94.40177373, 94.38733147, 94.32935596, 94.31297316000001, 94.27185757, 94.21031714, 94.17865895000001, 94.155814, 94.12752442, 94.17210308000001, 94.3051161, 94.37049684, 94.49105147, 94.59000234, 94.71180982000001, 94.84726905, 94.98310969, 95.11708095, 95.23249851000001, 95.35086952, 95.49013471, 95.59274907, 95.6903159, 95.76917677, 95.85176559000001, 95.98873876, 96.12517014, 96.2667368, 96.34254552, 96.4691226, 96.60357757000001, 96.72038309000001, 96.83910559, 96.94770207, 96.99852515, 97.10177856, 97.17007191, 97.22760416000001, 97.28711257, 97.40086396, 97.43212981, 97.53923681, 97.47417095, 97.53698453, 97.61594555, 97.63002918, 97.66363895, 97.68649476, 97.65989801, 97.72614530000001, 97.73078317, 97.72498263, 97.77438843, 97.82792222, 97.88235571000001, 97.92227668, 97.99600649000001, 97.99631542, 98.0027498, 97.98722563, 98.05106475, 98.07103755, 98.19383709, 98.29193081, 98.35610235, 98.41574411, 98.38220571, 98.48128813, 98.48581137, 98.54642323, 98.58513298, 98.55186045, 98.48282976, 98.37803658, 98.24041541, 98.16175535, 98.190719, 98.21627819, 98.28373119, 98.42406625, 98.47585313, 98.35487042, 98.38423963, 98.31816816000001, 98.19798209, 98.16403212, 98.12014225, 98.06943185, 98.19303771000001, 98.31551745, 98.34949856, 98.48376527, 98.59894774, 98.65808376, 98.64157975, 98.65044426, 98.67447351000001, 98.58109059, 98.44571862, 98.37139806, 98.42198799, 98.3846759, 98.42653752, 98.43061116, 98.40608195, 98.41908372, 98.42827475, 98.45]
lats_myanmar = [20.59685207, 20.48527765, 20.36253146, 20.26526894, 20.14908509, 20.03647085, 19.90893338, 19.82417925, 19.78600446, 19.83710899, 19.97753506, 19.968072929999998, 19.828921819999998, 19.68784041, 19.58738475, 19.45924615, 19.34172023, 19.22321641, 19.09793626, 18.97699003, 18.89566138, 18.81235715, 18.68616831, 18.702841279999998, 18.6960166, 18.59269432, 18.45336621, 18.33088398, 18.21504994, 18.10751801, 17.975384209999998, 17.83710494, 17.700431509999998, 17.57073633, 17.42398123, 17.29324113, 17.16977787, 17.02320388, 16.89829757, 16.75569997, 16.6146675, 16.48178124, 16.33800436, 16.19262861, 16.04812272, 15.916079759999999, 15.8853756, 15.76505663, 15.828244419999999, 15.79754487, 15.71803052, 15.668910539999999, 15.615503579999999, 15.64278171, 15.66424897, 15.598725199999999, 15.63815279, 15.74358277, 15.853940719999999, 15.975196429999999, 16.08579889, 16.13337581, 16.18369203, 16.21477723, 16.33937627, 16.3879363, 16.43688491, 16.52442467, 16.60974981, 16.69195779, 16.82381178, 16.775875969999998, 16.65669204, 16.521370859999998, 16.38928325, 16.374737409999998, 16.230790849999998, 16.17159949, 16.047871400000002, 15.91463435, 15.804063289999998, 15.65743513, 15.516031369999999, 15.37562801, 15.23515529, 15.108498959999999, 14.964803569999999, 14.81761485, 14.68466336, 14.55292429, 14.41721396, 14.27593108, 14.14992298, 14.04522381, 13.901903489999999, 13.77847435, 13.65481763, 13.508874989999999, 13.49157546, 13.514234349999999, 13.38397015, 13.25108981, 13.13265869, 13.03171879, 12.904378549999999, 12.783227539999999, 12.661019329999998, 12.51878708, 12.619976659999999, 12.70434592, 12.68297718, 12.57341231, 12.434958969999999, 12.36048288, 12.24031243, 12.25577878, 12.142099759999999, 12.099234269999998, 11.95494053, 11.850078089999998, 11.804265, 11.66100489, 11.524248179999999, 11.3880453, 11.366181039999999, 11.440917319999999, 11.543699479999999, 11.509645769999999, 11.55942078, 11.60335991, 11.48397393, 11.3608202, 11.219548929999998, 11.10704103, 11.05203489, 10.954721469999999, 10.84654436, 10.70404533, 10.569945639999998, 10.42445135, 10.315095099999999, 10.18580476, 10.076186479999999, 9.95]

# BEACHING
def MakeParticleSet(fieldset): # defining a function
        #class BeachableParticles(JITParticle): # defining a class of particle within this function - an instance of this class will be initialised through calling this function
        class BeachableParticles(ScipyParticle):
                beached = Variable('beached', dtype=np.int32, initial=0) # 'beached' is now an available variable of an oject of the class: 'BeachedParticles'
        return ParticleSet.from_list(fieldset, BeachableParticles, # this returns an instance of the BeachableParticles class
                                     #lon=[44, 45, 59, 68.5, 73, 65, 75, 79, 81, 89, 87, 99],
                                     #lat=[0, 12, 24, 22.5, 15, 12, 0, 7, 15, 20, 14, 5],
                                     lon = lons_myanmar,
                                     lat = lats_myanmar,
                                     repeatdt=timedelta(hours=1)) # 48,384 particles in 14 days

				      
# WINDAGE
windage = 0.01
filenames_wind = {'U_wind': input_velocities_filepath + "windage_u_avgd_interpolated_July2020_hourly.nc",
                 'V_wind': input_velocities_filepath + "windage_v_avgd_interpolated_July2020_hourly.nc"}
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
filenames_stokes = {'U_stokes': input_velocities_filepath + "stokes_u_avgd_interpolated_July2020_hourly_fromcoarse.nc",
                 'V_stokes': input_velocities_filepath + "stokes_v_avgd_interpolated_July2020_hourly_fromcoarse.nc"}
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

output_file = pset.ParticleFile(name=output_filename, outputdt=timedelta(minutes=60))

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


