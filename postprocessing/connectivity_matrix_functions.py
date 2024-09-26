import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rtree
import xarray as xr
import fiona
import shapely.geometry
import csv


def make_polygons_geojson_file(input_shape_file, min_lon, max_lon, min_lat, max_lat, output_polygon_filename):
    '''converts ICES rectangle data from a shape file into a geojson file that can be used by connectivity matrix analysis functions'''
    
    #read and extract info from shape file
    ices_shp = gpd.read_file(input_shape_file)
    ices_sample = ices_shp.loc[ices_shp.bounds.minx>min_lon].loc[ices_shp.bounds.maxx<max_lon].loc[ices_shp.bounds.miny>min_lat].loc[ices_shp.bounds.maxy<max_lat]
    print(ices_sample)
    # write geojson file using ICES rectangle coordinates
    ii=0
    outfile = open(output_polygon_filename,'w+')
    outfile.write('{\n')
    outfile.write('"type": "FeatureCollection",\n')
    outfile.write('"name": "ICES rectangle polygons",\n')
    outfile.write('"features": [\n')
    
    for n in range(len(ices_sample)):
        poly = ices_sample.bounds.iloc[n] 
        ii+=1
        outfile.write('{ "type": "Feature", "properties": { "name":')  
        outfile.write(' "#{}"'.format(ii))
        outfile.write(' }, "geometry": { "type": "Polygon", "coordinates": [ [ ')
        outfile.write('[{},{}], [{},{}], [{},{}], [{},{}], [{},{}] ] ]'.format(poly.minx, poly.miny, poly.maxx, poly.miny, poly.maxx, poly.maxy, poly.minx, poly.maxy, poly.minx, poly.miny))
    
        if n<len(ices_sample)-1:
            outfile.write(' } },\n')
        else:
            outfile.write(' } }\n')
            outfile.write(']\n')
            outfile.write('}\n')

    outfile.close()
    
    return outfile


# Create an rtree index based on polygon bounds to
# determine if shapely.geometry.point().within()
# At the same time a list is created with the 
# shape's names.
def index_polygons(polygons, field):

    idx = rtree.index.Index()
    names = []
    for pos, poly in enumerate(polygons):
        idx.insert(pos, shapely.geometry.shape(poly['geometry']).bounds)
        names.append( poly['properties'][field] ) 

    return idx, names


def create_shp_index(polygons_fname, polygons_field):

    # Create the R-tree index and store the features in it (bounding box)
    polygons = fiona.open(polygons_fname)
    polygons_idx, polygons_names = index_polygons(polygons, polygons_field)
    print('File',polygons_fname, ' has ',len(polygons_names),' polygons', flush=True)

    return polygons, polygons_idx, polygons_names


# Find 1st polygon containing point
# returns polygon id or None if no match
# Uses Rtree.intersection to accelerate the search
def find_poly(polygons,polyind,coords):
    
    for polyid in polyind.intersection( coords ):
        point = shapely.geometry.Point( coords )
        if point.within( shapely.geometry.shape( 
                polygons[polyid]['geometry'] ) ):
            return polyid

    return None


def calc_connectivity_lh(input_particle_file, polygons_start_filename, 
                      polygons_end_filename,
                      start_polygons, start_polygons_idx,
                      end_polygons, end_polygons_idx):
    
    trajDS = xr.load_dataset(input_particle_file)
    
    polygons_start_gpd = gpd.read_file(polygons_start_filename)
    polygons_end_gpd = gpd.read_file(polygons_end_filename)
    n_start_polygons = len(polygons_start_gpd['geometry'][:])
    n_end_polygons = len(polygons_end_gpd['geometry'][:])
    connec_mat=np.zeros([n_start_polygons, n_end_polygons])

    for iparticle in range( trajDS.dims['traj'] ):
        print('iparticle: ', iparticle)
        if any(trajDS.variables['beached'][iparticle,:]==1):
            # Deal with Origin first
            start_loc = (trajDS['lon'].isel(traj=iparticle,obs=0).values, 
                         trajDS['lat'].isel(traj=iparticle,obs=0).values)
#            print('start_loc:',start_loc)
#            print('start_polygons_idx:',start_polygons_idx)
            start_id = find_poly(polygons=start_polygons,
                                 polyind=start_polygons_idx, 
                                 coords=start_loc )
#            print(f'start_id is {start_id}')

    
            if start_id is None:
                print("WARNING particle {0} originating at ({1:.3f},{2:.3f}) is not in a polygon".format(iparticle,start_loc[1],start_loc[0]), flush=True)
                continue

            lons = trajDS['lon'].isel(traj=iparticle).values
            lats = trajDS['lat'].isel(traj=iparticle).values

            for itime in range(len(lons)):
#                print('itime:', itime)
                if not(np.isnan( lons[itime])):
                    end_polygon_id = find_poly(
                                        polygons = end_polygons,
                                        polyind = end_polygons_idx,
                                        coords = (lons[itime],lats[itime])
                    )

            if end_polygon_id is None:
                print("WARNING particle {0} does not end in a polygon".format(iparticle), flush=True)
                continue
    
            print('start_id: ', start_id)
            print('end_polygon_id: ', end_polygon_id)
        

            connec_mat[start_id, end_polygon_id] = connec_mat[start_id, end_polygon_id] + 1
        
            connec_mat

    return connec_mat



def write_to_file(out_fname, start_polygons_names, 
                  end_polygons_names, connec_mat):
    
    with open(out_fname, 'w',newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['']+end_polygons_names)
        for irow in range(connec_mat.shape[0]):
            spamwriter.writerow(
                [start_polygons_names[irow]]+list(connec_mat[irow])
            )



def plot_conn_mat(input_conn_mat_file, output_figname):
    conn_mat_df = pd.read_csv(input_conn_mat_file, index_col=0)
    conn_mat_df.replace(0.0, np.nan, inplace=True) # replace all zeros with NaNs, inplace=True changes it in the dataframe itself not in the variable created here. 
    polygon_names_list = list(conn_mat_df) # this prints out the column headings of the dataframe. For connectivity marices, the column and row headings are the same so I don't need to extract the row headings (which I think is more complicated...). I can just use the column headings as xticklabels and yticklabels
    print(polygon_names_list)
    #create figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), dpi=300)
#    m1 = ax.matshow(conn_mat_df, cmap='spring') #, vmin=-2, vmax=3) # mappable content
    m1 = ax.matshow(conn_mat_df, cmap='cool')#, vmin=0, vmax=1) # mappable content

    for (ii, jj), z in np.ndenumerate(conn_mat_df):
        if ~np.isnan(z):
            ax.text(jj, ii, '{:0f}'.format(z), ha='center', va='center')
    
    # ax.set_title('title ' + str(1), fontsize=18)
    ax.set_ylabel('Source location', fontsize=15)
    ax.set_xlabel('Settle location', fontsize=15)
    #ax.set_xticklabels(['']+polygon_names_list)
    #ax.set_yticklabels(['']+polygon_names_list)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(polygon_names_list)), labels=polygon_names_list)
    ax.set_yticks(np.arange(len(polygon_names_list)), labels=polygon_names_list)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")
    # m1.set_clim(0,max(np.amax(speed_subset_start),np.amax(speed_subset_end))) # same as setting vmin/vmax limits in countourf() or pcolormesh() line of code
    
    cbar = fig.colorbar(m1, shrink=0.82)
    # cbar.set_label('m/s', fontsize=15)
    
    fig.savefig(output_figname, bbox_inches='tight', facecolor='white', transparent=False)


def plot_conn_mat_norm(input_conn_mat_file, output_figname):
    conn_mat_df = pd.read_csv(input_conn_mat_file, index_col=0)
    conn_mat_df.replace(0.0, np.nan, inplace=True) # replace all zeros with NaNs, inplace=True changes it in the dataframe itself not in the variable created here.
    polygon_names_list = list(conn_mat_df) # this prints out the column headings of the dataframe. For connectivity marices, the column and row headings are the same so I don't need to extract the row headings (which I think is more complicated...). I can just use the column headings as xticklabels and yticklabels
    print(polygon_names_list)

    #create figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), dpi=300)
#    m1 = ax.matshow(conn_mat_df, cmap='spring') #, vmin=-2, vmax=3) # mappable content
    m1 = ax.matshow(conn_mat_df, cmap='cool', vmin=0, vmax=1) # mappable content

    for (ii, jj), z in np.ndenumerate(conn_mat_df):
        if ~np.isnan(z):
            ax.text(jj, ii, '{:0.2f}'.format(z), ha='center', va='center')

    # ax.set_title('title ' + str(1), fontsize=18)
    ax.set_ylabel('Source location', fontsize=15)
    ax.set_xlabel('Settle location', fontsize=15)
    #ax.set_xticklabels(['']+polygon_names_list)
    #ax.set_yticklabels(['']+polygon_names_list)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(polygon_names_list)), labels=polygon_names_list)
    ax.set_yticks(np.arange(len(polygon_names_list)), labels=polygon_names_list)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")
    
    # m1.set_clim(0,max(np.amax(speed_subset_start),np.amax(speed_subset_end))) # same as setting vmin/vmax limits in countourf() or pcolormesh() line of code

    cbar = fig.colorbar(m1, shrink=0.82)
    # cbar.set_label('m/s', fontsize=15)

    fig.savefig(output_figname, bbox_inches='tight', facecolor='white', transparent=False)


def plot_conn_mat_norm_noAN(input_conn_mat_file, output_figname):
    conn_mat_df = pd.read_csv(input_conn_mat_file, index_col=0)
    conn_mat_df.replace(0.0, np.nan, inplace=True) # replace all zeros with NaNs, inplace=True changes it in the dataframe itself not in the variable created here.
    polygon_names_list_source = list(conn_mat_df) # need to remove Andaman and Nicobar
    polygon_names_list_source = [polygon_names_list_source[x] for x in [0,1,2,3,4,5] ]
    #polygon_names_list_source = [polygon_names_list_source[x] for x in [0,1,2,3,4,5,6,7,10] ]
    polygon_names_list_sink = list(conn_mat_df) # this prints out the column headings of the dataframe. For connectivity marices, the column and row headings are the same so I don't need to extract the row headings (which I think is more complicated...). I can just use the column headings as xticklabels and yticklabels
    print(polygon_names_list_source)
    print(polygon_names_list_sink)

    #create figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), dpi=300)
#    m1 = ax.matshow(conn_mat_df, cmap='spring') #, vmin=-2, vmax=3) # mappable content
    m1 = ax.matshow(conn_mat_df, cmap='Blues', vmin=0, vmax=1) # mappable content

    for (ii, jj), z in np.ndenumerate(conn_mat_df):
        if ~np.isnan(z):
            ax.text(jj, ii, '{:0.2f}'.format(z), ha='center', va='center')

    # ax.set_title('title ' + str(1), fontsize=18)
    ax.set_ylabel('Source location', fontsize=15)
    ax.set_xlabel('Settle location', fontsize=15)
    #ax.set_xticklabels(['']+polygon_names_list)
    #ax.set_yticklabels(['']+polygon_names_list)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(polygon_names_list_sink)), labels=polygon_names_list_sink)
    ax.set_yticks(np.arange(len(polygon_names_list_source)), labels=polygon_names_list_source)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")

    # m1.set_clim(0,max(np.amax(speed_subset_start),np.amax(speed_subset_end))) # same as setting vmin/vmax limits in countourf() or pcolormesh() line of code

    cbar = fig.colorbar(m1, shrink=0.69)
    # cbar.set_label('m/s', fontsize=15)

    fig.savefig(output_figname, bbox_inches='tight', facecolor='white', transparent=False)



def plot_conn_mat_diff(input_conn_mat_file1, input_conn_mat_file2, output_diff_figname,vmin,vmax):
    conn_mat_df1 = pd.read_csv(input_conn_mat_file1, index_col=0)
    conn_mat_df2 = pd.read_csv(input_conn_mat_file2, index_col=0)
    conn_mat_df = conn_mat_df1 - conn_mat_df2
    conn_mat_df.replace(0.0, np.nan, inplace=True) # replace all zeros with NaNs, inplace=True changes it in the dataframe itself not in the variable created here.
    polygon_names_list = list(conn_mat_df1) # this prints out the column headings of the dataframe. For connectivity marices, the column and row headings are the same so I don't need to extract the row headings (which I think is more complicated...). I can just use the column headings as xticklabels and yticklabels
    print(conn_mat_df)
    #create figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), dpi=300)
#    m1 = ax.matshow(conn_mat_df, cmap='spring') #, vmin=-2, vmax=3) # mappable content
    m1 = ax.matshow(conn_mat_df, cmap='PRGn', vmin=vmin, vmax=vmax) # mappable content

    # ax.set_title('title ' + str(1), fontsize=18)
    ax.set_ylabel('Source location', fontsize=18)
    ax.set_xlabel('Settle location', fontsize=18)
    #ax.set_xticklabels(['']+polygon_names_list)
    #ax.set_yticklabels(['']+polygon_names_list)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(polygon_names_list)), labels=polygon_names_list, fontsize=18)
    ax.set_yticks(np.arange(len(polygon_names_list)), labels=polygon_names_list, fontsize=18)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")

    # m1.set_clim(0,max(np.amax(speed_subset_start),np.amax(speed_subset_end))) # same as setting vmin/vmax limits in countourf() or pcolormesh() line of code

    cbar = fig.colorbar(m1, shrink=0.82)
    cbar.ax.tick_params(labelsize=18)
    # cbar.set_label('m/s', fontsize=15)

    fig.savefig(output_diff_figname, bbox_inches='tight', facecolor='white', transparent=False)
