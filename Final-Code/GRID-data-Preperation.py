# %%
import numpy as np
import pandas as pd
import datetime 
import logging
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from datetime import timedelta
import os
import pickle
# import tqdm 
from datetime import timedelta

# %%
df1 = pd.read_csv("/storage3/DSIP/Demeter/ICE/I_ICE_CleanINPUT_Orbits.csv") # This is a file where i have the names of all day time nad night time orbits 
logging.warning(' df')
# df1 = df1[12979:17461]
# df1 = df1[8193:12979]
# df1 = df1[17461:]
# %%
nf = []
for filename in df1['DownOrbits']:
    fn = filename
    parts = fn.split('.')
    sp_fn = parts[0] + "_full.pkl"
    nf.append(sp_fn)
 
df1.loc[:, 'sp_filename'] = nf

logging.warning('sp_filename done')
# %%

pickle_folder = "/storage3/DSIP/Demeter/ASCII-ICE" #folder of all ASCII files

# %%


def create_grid(width, depth):

    min_lat, max_lat = -90, 90
    min_lon, max_lon = -180, 180
    
    num_lat_cells = int((max_lat - min_lat) / depth)
    num_lon_cells = int((max_lon - min_lon) / width)

    grid = np.empty((num_lat_cells, num_lon_cells), dtype=object)
    
    # Assign IDs to each grid cell
    for lon_idx in range(num_lon_cells):
        for lat_idx in range(num_lat_cells):
            lat_start = min_lat + lat_idx * depth
            lat_end = lat_start + depth
            lon_start = min_lon + lon_idx * width
            lon_end = lon_start + width
            grid[lat_idx, lon_idx] = f'G{lat_idx * num_lon_cells + lon_idx + 1}: (({lon_start},{lon_end}),({lat_start},{lat_end}))'
            # print(lat_idx,num_lon_cells,lon_idx,lat_idx * num_lon_cells + lon_idx + 1)
            # print(grid[lat_idx, lon_idx])
    return grid

# Example usage
grid = create_grid(20, 20)
# print(grid)


# %%


def create_polygon(lat, lon, width):
   
    lat_delta = width / 2
    lon_delta = width / 2
    latitudes = [lat - lat_delta, lat + lat_delta, lat + lat_delta, lat - lat_delta]
    longitudes = [lon - lon_delta, lon - lon_delta, lon + lon_delta, lon + lon_delta]

    
    polygon = Polygon(zip(longitudes, latitudes))

    return polygon

# %%



def calculate_average_spectrum(df, width, delta_t, start_time=None):
    df['geoc_lat'] = df['geoc_lat'].astype(float)
    df['geoc_long'] = df['geoc_long'].astype(float)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['geoc_long'] = df['geoc_long'].apply(pd.to_numeric, errors='coerce')
    df['geoc_long'] = df['geoc_long'].apply(lambda x: x - 360 if x > 180 else x)
    result = {}  
    Spectrum =[]
    df = df.sort_values(by='datetime')  

    if start_time is None:
        start_time = df.iloc[0]['datetime']

    # Fb for ICE
    bands = [
        (1 * 19.53, 2 * 19.53),
        (15 * 19.53, 19 * 19.53),
        (19 * 19.53, 24 * 19.53),
        (24 * 19.53, 31 * 19.53),
        (31 * 19.53, 39 * 19.53),
        (39 * 19.53, 49 * 19.53),
        (49 * 19.53, 61 * 19.53),
        (61 * 19.53, 77 * 19.53),
        (77 * 19.53, 97 * 19.53),
        (97 * 19.53, 122 * 19.53),
        (122 * 19.53, 154 * 19.53)
    ]

    #fb for IMSC
    # bands = [
    #     (1 * 19.53, 2 * 19.53),
    #     (15 * 19.53, 20 * 19.53),
    #     (20 * 19.53, 25 * 19.53),
    #     (25 * 19.53, 32 * 19.53),
    #     (32 * 19.53, 40 * 19.53),
    #     (40 * 19.53, 51 * 19.53)
    # ]
    min_freq = 19.53
    max_freq = 20000
   
    while start_time:
        t = start_time  

        
        row_with_start_time = df[df['datetime'] == start_time]

        if not row_with_start_time.empty:
            lat = row_with_start_time.iloc[0]['geoc_lat']
            lon = row_with_start_time.iloc[0]['geoc_long']

            polygon = create_polygon(lat, lon, width)

            fdf = df[df.apply(lambda x: Point(x['geoc_long'], x['geoc_lat']).within(polygon), axis=1)]

            if not fdf.empty:
                sp0 = np.array([np.array(x) for x in fdf['spectrum0']]).astype(float)
                sp1 = np.array([np.array(x) for x in fdf['spectrum1']]).astype(float)
                sp = np.empty((sp0.shape[0], sp0.shape[1]*2))
                for i in range(sp0.shape[0]):
                    sp[i] = np.concatenate((sp0[i], sp1[i]))
                sp = sp.reshape((sp0.shape[0]*2, 1024))
                # print('sp0',sp0)
                # print(len(sp0))
                # print('sp1',sp1)
                # print(len(sp1))
                # print('sp')
                # print(sp)
                # plt.imshow(sp0.T)
                # plt.show()
                # plt.imshow(sp1.T)
                # plt.show()
                # plt.imshow(sp.T)
                # plt.show()
                for band_id, band in enumerate(bands,1):
                    min_freq_band, max_freq_band = band

                    start_index = int(min_freq_band / (max_freq / 1024))
                    end_index = int(max_freq_band / (max_freq / 1024))
                    
                    sp_range = sp[:, start_index:end_index].flatten()
                    # print(len(sp_range))
                    

                    Q1 = np.percentile(sp_range, 25)
                    Q2 = np.percentile(sp_range, 50)                     
                    Q3 = np.percentile(sp_range, 75)
                    # P12_5 = np.percentile(sp_range, 12.5)
                    # P87_5 = np.percentile(sp_range, 87.5)
                    # maximum = max(sp_range)
                    
                    # Filter the data to include only the central 75%
                    # Q75 = sp_range[(sp_range > P12_5) & (sp_range < P87_5)]
                    # print(len(Q75))

                    grid = create_grid(20, 20).reshape(-1)
                
                    best_grid_id = None
                    max_intersection = 0

                   
                    for g_id, g in enumerate(grid,1):
                        full_grid_id = f'G{g_id}'
                        
                        lon_range, lat_range = g.split(': ((')[1].strip(')').split('),(')
                        lon_start, lon_end = map(float, lon_range.split(','))
                        lat_start, lat_end = map(float, (lat_range.split(',')))

                       
                        grid_polygon = Polygon([(lon_start, lat_start), (lon_start, lat_end), (lon_end, lat_end), (lon_end, lat_start)])

                        
                        polygon_area = polygon.area
                        intersection_area = grid_polygon.intersection(polygon).area
                        percent_intersection = (intersection_area / polygon_area) * 100

                        
                        if percent_intersection > max_intersection:
                            max_intersection = percent_intersection
                            best_grid_id = full_grid_id
                    #         print(best_grid_id, max_intersection)
                    # print('the best grid ',best_grid_id )
                    if best_grid_id is not None:
                        if best_grid_id not in result:
                            result[best_grid_id] = []
                        # Store statistics for the current band in the result dictionary
                        result[best_grid_id].append({
                            f"fb_{band_id}": {'datetime': t,
                                                    'Q1': Q1,
                                                    'Q2':Q2,
                                                    'Q3':Q3,
                                                     'lat':lat,
                                                     'lon':lon
                                    
                                                        }
                        })

        
        next_time = start_time + timedelta(minutes=delta_t)
        next_time_row = df[df['datetime'] >= next_time]
        if not next_time_row.empty:
            # Update start_time
            start_time = next_time_row.iloc[0]['datetime']
        else:
           
            break

    return result
#%%

#%%

logging.warning('moving to the loop of fn')
# %%
aggregate_result_dict = {}

# Iterate through each pickle file
for file in df1['sp_filename']:
    # Load DataFrame from pickle file
    df = pd.read_pickle(os.path.join(pickle_folder, file))

    # Apply calculate_average_spectrum function
    result = calculate_average_spectrum(df, 10, 2, start_time=None)

    # Merge result with aggregate_result_dict
    for key, value in result.items():
        if key not in aggregate_result_dict:
            aggregate_result_dict[key] = []
        aggregate_result_dict[key].extend(value)


with open("/storage3/DSIP/Demeter/Newdataset/Down_Orbits-Q123-ICE_2026.pkl", "wb") as f:
    pickle.dump(aggregate_result_dict, f)


print("Aggregated results saved to aggregate_result_full.pkl")

# %%



