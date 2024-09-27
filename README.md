# Code to accompany paper submitted to Ocean Science ()

## Preprocessing
OceanParcels run scripts and kernels.
Scripts are separated by country release locations and season. Monsoon season runs from 1st June 2018 - 30th September 2019, post-monsoon season runs from 1st October 2018 - 30th September 2019, pre-monsoon season runs from 1st February 2019 - 30th September 2019. Particles are released daily for four months in each case. Two different forcings were used - ROMS (1/48° resolution, no data assimilation) and CMEMS (1/12° resolution, data assimilation).
### Sensitivity studies
Run scripts to compare hourly versus daily forcing. Uses same kernels as in main run scripts.
### Validation
Run scripts to release 100 particles per week in same location as an undrogued drifter (model output compared with Global Drifter Programme trajectories). Uses same kernels as in main run scripts.

## Postprocessing
Scripts to populate connectivity matrices from model output (connectivity_matrix_analysis.ipynb, uses connectivity_matrix_functions.py), to calculate particles still afloat at the end of a simulation or that left the domain during the simulation (quantify_escaped_particles.ipynb), and to calculate the distances between observed and simulated trajectories for model validation (sep_dist_calc.ipynb). 

Code to create plots and animations from the paper and supplementary materials is also in here (make_paper_plots.ipynb, parcels_animation_full_year_ROMS.py, parcels_animation_full_year_CMEMS.py).

Notes: 

* connectivity_matrix_functions.py were adapted from functions written by Tiago Silva found [here](https://github.com/CefasRepRes/gitm-utils?tab=readme-ov-file#connectivity_matrix). This is a CefasRepRes private repository. 

* Calculations of separation distances between observed and simulated drifters are found at the bottom of the sep_dist_calc.ipynb script. Identifying the points of the drifter/particle trajectories to compare was done by hand via a combination of Excel and Python so the script is rather large and isn't really code so much as lots of copied and pasted lines to be commented and uncommented.    

