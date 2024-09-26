from parcels import FieldSet, ParticleSet, Variable, JITParticle, ScipyParticle, Field, VectorField, plotTrajectoriesFile, ParcelsRandom
from parcels.tools.converters import Geographic, GeographicPolar
import numpy as np
import math
from math import fabs
from datetime import timedelta
from operator import attrgetter
from all_kernels import AdvectionRK4, DiffusionUniformKh, Windage, StokesDrift, BeachedStatusCheck, BeachedDelete

output_filename="/gpfs/home/rpe16nbu/projects/ocpp_mo1/data/output/paper_data/BoB_India_uniform_ROMS_daily_Jun2018-Sept2019_monsoon.nc"

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

lons_india = [84.84714845, 84.95417304, 85.05906622, 85.17990843, 85.28245046, 85.397487, 85.53244544, 85.66928233, 85.80722822, 85.94562129, 86.08853421, 86.22877569, 86.36195667, 86.46856255999998, 86.54213869999998, 86.64892043, 86.78213481999998, 86.82116297999998, 86.84583306, 86.96283022, 87.06131682999998, 87.11781175, 87.00718322, 86.95077757, 86.89562411999998, 86.96540478, 87.06976446999998, 87.17567019, 87.31805056999998, 87.45846448, 87.59515417, 87.74062297, 87.86858051, 87.96123547, 88.00873962, 88.10664479, 88.21233698, 88.34122983, 88.45192310999998, 88.5754112, 88.65470752999998, 88.78688169, 88.91841499999998, 89.00149983999998, 89.13704391999998, 89.14678927, 79.93043740999998, 79.92317744999998, 79.91591108999998, 79.90927734, 79.91049830999998, 79.91108061, 79.88609137999998, 79.86861173, 79.81008613999998, 79.83564054999998, 79.86831164, 79.92438346, 80.00299416999998, 80.07499799, 80.16595644999998, 80.22343454, 80.26086241, 80.29355936, 80.31676208, 80.35235457, 80.38281761999998, 80.38577059, 80.34628098999998, 80.30507812, 80.3082452, 80.25239376, 80.19038318, 80.19250961, 80.21973298, 80.22789457, 80.21383613, 80.16231053999998, 80.137237, 80.10910019, 80.14102739, 80.16545444, 80.25745033, 80.29980011, 80.3839367, 80.51009555999998, 80.65320316, 80.74884294, 80.83096652, 80.97684024999998, 81.05939531999998, 81.139317, 81.20708713, 81.25862080999998, 81.34365112, 81.48168554, 81.62526547, 81.76393505, 81.90117531999998, 82.03841842999998, 82.17176487, 82.28840176999998, 82.37427904, 82.39279870999998, 82.38525874, 82.35163814, 82.4517637, 82.56235426, 82.68228379, 82.81132671999998, 82.94880865999998, 83.07914809, 83.21141627, 83.32822097, 83.41459605999998, 83.50132978, 83.60031935, 83.71563997, 83.84068859, 83.97415436999998, 84.10604101, 84.20132056999998, 84.29916020999998, 84.41051394999998, 84.50460287999998, 84.59743928999998, 84.67823789, 84.77476582, 77.05516740999998, 77.16113036999998, 77.29248074, 77.43085189999998, 77.5519883, 77.66011446, 77.80033519, 77.92140417, 78.04839269999998, 78.14230836, 78.18932103, 78.21838431, 78.24040969, 78.31790277, 78.43655355, 78.57809281999998, 78.71780296999998, 78.86175355, 78.98595697, 79.07173437999998, 78.97876152, 79.00426686999998, 79.09912174, 79.18849423999998, 79.27108108999998, 79.30517193, 79.36430403999998, 79.49104973999998, 79.63768025999998, 79.78240999, 79.91779418]
lats_india = [19.073992859999997, 19.17513682, 19.27841259, 19.35895626, 19.46458366, 19.556254789999997, 19.60707519, 19.658152349999998, 19.70855098, 19.754669529999997, 19.78894789, 19.83181067, 19.89358591, 19.98142524, 20.108513709999997, 20.200536409999998, 20.26340209, 20.3854361, 20.510249199999997, 20.599744079999997, 20.70151128, 20.831114619999997, 20.908695209999998, 21.04426328, 21.180551819999998, 21.30927405, 21.4072192, 21.50074785, 21.5314005, 21.57060262, 21.61573248, 21.63890559, 21.7046188, 21.74367868, 21.625042909999998, 21.545507949999998, 21.49592224, 21.56091295, 21.53442059, 21.53992143, 21.58803129, 21.536727759999998, 21.52326329, 21.603164279999998, 21.649449469999997, 21.67188297, 10.410387069999999, 10.55751102, 10.70463466, 10.85176106, 10.998950129999999, 11.146137439999999, 11.29095278, 11.43721497, 11.569837779999999, 11.7148302, 11.858464179999999, 11.987890109999999, 12.111812899999999, 12.23903826, 12.350783089999998, 12.484573919999999, 12.62684358, 12.77047183, 12.915829389999999, 13.058005929999998, 13.20211404, 13.34623222, 13.48756965, 13.62522817, 13.77241495, 13.902901539999998, 14.036184819999999, 14.18075784, 14.3251712, 14.47035976, 14.60387864, 14.74166428, 14.88664951, 15.030884879999999, 15.16798491, 15.31077974, 15.42258363, 15.5634699, 15.679138949999999, 15.747757409999998, 15.7809675, 15.708209239999999, 15.617544919999998, 15.63801488, 15.741615099999999, 15.862485309999999, 15.990691239999999, 16.12749412, 16.24308962, 16.272610489999998, 16.24851349, 16.25151656, 16.30366123, 16.35579715, 16.41835148, 16.49502072, 16.591430969999998, 16.73701799, 16.85719826, 16.96911582, 17.07715765, 17.174243439999998, 17.257779, 17.32837988, 17.37714303, 17.445756449999998, 17.51051466, 17.59628353, 17.71560456, 17.83264635, 17.93573975, 18.02613159, 18.10060566, 18.16255347, 18.22807366, 18.33363211, 18.44226771, 18.53869649, 18.65110851, 18.76546871, 18.888553379999998, 18.999821179999998, 8.28748287, 8.18563802, 8.122878159999999, 8.08373068, 8.03020016, 8.12691211, 8.16543408, 8.248869690000001, 8.32351605, 8.43414173, 8.56971587, 8.71329207, 8.85877029, 8.96810792, 9.05525016, 9.09170063, 9.13478193, 9.16602928, 9.240709540000001, 9.41, 9.459619700000001, 9.60075786, 9.71132942, 9.822195650000001, 9.94338466, 10.07920169, 10.19706562, 10.24876032, 10.23470133, 10.20814136, 10.21514832]

# BEACHING
def MakeParticleSet(fieldset): # defining a function
        #class BeachableParticles(JITParticle): # defining a class of particle within this function - an instance of this class will be initialised through calling this function
        class BeachableParticles(ScipyParticle):
                beached = Variable('beached', dtype=np.int32, initial=0) # 'beached' is now an available variable of an oject of the class: 'BeachedParticles'
        return ParticleSet.from_list(fieldset, BeachableParticles, # this returns an instance of the BeachableParticles class
                                     #lon=[44, 45, 59, 68.5, 73, 65, 75, 79, 81, 89, 87, 99],
                                     #lat=[0, 12, 24, 22.5, 15, 12, 0, 7, 15, 20, 14, 5],
                                     lon = lons_india,
                                     lat = lats_india,
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


