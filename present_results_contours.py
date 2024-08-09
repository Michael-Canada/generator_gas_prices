import pandas as pd
import matplotlib.pyplot as plt

distance = 700
generators_with_close_hubs = pd.read_csv(f"generators_with_close_hubs_{distance}.csv")

# the columns of the DataFrame 'generators_with_close_hubs' are: 'plant_id', 'quantity', 'fuel_cost', 'plant_name', 'latitude',
    # 'longitude', 'closest_gas_hub_symbol', 'distance_to_closest_gas_hub',
    # 'closest_gas_hub_name', 'closest_gas_hub_lat', 'closest_gas_hub_lon',
    # 'current_supplier_hub_symbol', 'distance_to_current_supplier',
    # 'current_supplier__hub_name', 'current_supplier__hub_lat',
    # 'current_supplier__hub_lon', 'current_supplier__Pvalue',
    # 'dollar_saving_by_choosing_current_supplier',
    # 'percent_saving_by_choosing_current_supplier'
    

import folium
import pandas as pd

NG_hub_loc_info = pd.read_parquet('/Users/michael.simantov/Documents/generator_gas_prices/ng_hub_definition_parquet.parquet')

# Initialize the map at a central point
m = folium.Map(location=[37.0902, -95.7129], zoom_start=5)

# Iterate over the rows of the DataFrame
for index, row in NG_hub_loc_info.iterrows():
    # Create a marker for each hub
    folium.Marker(
        [row['latitude'], row['longitude']],  # Use the hub's latitude and longitude
        icon=folium.Icon(color='green', icon='square', icon_size=(24, 24)),  # Customize the icon
        popup=row['hub_name'],  # Use the hub's name for the popup
    ).add_to(m)  # Add the marker to the map

# Assuming 'data' now includes a 'current_supplier__hub_name' column
data = generators_with_close_hubs[['plant_name','latitude', 'longitude', 'current_supplier__hub_lat', 'current_supplier__hub_lon', 'current_supplier__hub_name', 'closest_gas_hub_lat', 'closest_gas_hub_lon','percent_saving_by_choosing_current_supplier']]

# # Display the map
# m.save(f'map_{distance}.html')

# print(18)



################################# CREATE DATA FOR CONTOURS ###########################################################################
import numpy as np
from scipy.interpolate import griddata
import json
from matplotlib.colors import to_hex
import duckdb

def get_hub_gas_prices():
    dt_str = '2023-'
    benchmark = duckdb.read_parquet("/Users/michael.simantov/Documents/generator_gas_prices/*.parquet", filename=True)
    hub_prices = duckdb.sql(f"SELECT * FROM benchmark WHERE filename LIKE '%{dt_str}%'").to_df()
    hub_prices["month"] = hub_prices.filename.map(lambda f: int(f.split("/")[-1].split(".")[0].split('-')[1]))
    # hub_prices["timestamp"] = hub_prices.case.map(df_cases.case_timestamp)

    hub_prices = hub_prices.drop(columns=["open","high","low","close","volume","open_interest","published_timestamp",'parent_dataset', 'parent_environment',
        'parent_location', 'parent_version'])
    # hub_prices = hub_prices.set_index("month")

    return hub_prices

hub_gas_prices = get_hub_gas_prices()
hub_gas_prices = hub_gas_prices[hub_gas_prices.price_symbol.isin(NG_hub_loc_info.mv_symbol)]
hub_gas_prices_per_month = hub_gas_prices.groupby(["month","price_symbol"]).agg({"mid_point": "mean"}).reset_index()
hub_gas_prices_pivot = hub_gas_prices_per_month.pivot(index="month", columns="price_symbol", values="mid_point")

average_gas_price_per_hub = hub_gas_prices_pivot.mean(axis=0).rename("average_gas_price")
NG_hub_loc_info = pd.merge(NG_hub_loc_info, average_gas_price_per_hub, left_on="mv_symbol", right_on="price_symbol")

# remove rows from NG_hub_loc_info if NG_hub_loc_info.hub_name is not in data.current_supplier__hub_name
NG_hub_loc_info = NG_hub_loc_info[NG_hub_loc_info.hub_name.isin(data.current_supplier__hub_name)]



# Step 1: Prepare data for interpolation
points = NG_hub_loc_info[['longitude', 'latitude']].values
values = NG_hub_loc_info['average_gas_price'].values

# Create a grid to interpolate onto. Increase the grid resolution by specifying a finer step size
grid_x, grid_y = np.mgrid[
    min(points[:,0]):max(points[:,0]):complex(0, 1500),  # 1500 points along longitude
    min(points[:,1]):max(points[:,1]):complex(0, 1500)   # 1500 points along latitude
]

# grid_x, grid_y = np.mgrid[min(points[:,0]):max(points[:,0]), min(points[:,1]):max(points[:,1])]

# Interpolate gas prices onto the grid
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

# Step 2: Generate contour data using matplotlib
plt.figure()
num_levels = 100  # Example: 100 contour levels for more granularity
CS = plt.contour(grid_x, grid_y, grid_z, levels=num_levels, cmap='seismic', vmin=grid_z.min(), vmax=grid_z.max())

############################### CREATE CONTOURS #############################################################################

min_val = NG_hub_loc_info.average_gas_price.min()
max_val = NG_hub_loc_info.average_gas_price.max()
ranges = np.round(np.linspace(min_val, max_val, 6), 2)

if True:
    import matplotlib

    geojson_features = []
    colormap = plt.cm.get_cmap('viridis')
    norm = plt.Normalize(vmin=CS.levels.min(), vmax=CS.levels.max())

    for i, contour in enumerate(CS.collections):
        for path in contour.get_paths():
            v = path.vertices
            coords = np.array(v).tolist()
            level = CS.levels[i]
            color = matplotlib.colors.rgb2hex(colormap(norm(level)))
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "stroke": color,
                    "stroke-width": 2,
                    "fillOpacity": 1,
                    "level": level
                },
            }
            geojson_features.append(feature)

    geojson_contours = {
        "type": "FeatureCollection",
        "features": geojson_features,
    }

    # # Create a Folium map
    # m = folium.Map(location=[0, 0], zoom_start=2)

    # Add the GeoJSON to the Folium map
    folium.GeoJson(
        geojson_contours,
        style_function=lambda x: {
            'color': x['properties']['stroke'],
            'weight': x['properties']['stroke-width'],
            'fillColor': x['properties']['stroke'],
            'opacity': x['properties']['fillOpacity'],
        }
    ).add_to(m)

    if True:
        legend_html = f'''
        <div style="position: fixed; 
        bottom: 50px; left: 50px; width: 200px; height: 180px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        ">&nbsp; Contour Values Legend <br>
        <!-- Color Box for Lowest Value -->
        <svg width="20" height="20">
        <rect width="20" height="20" style="fill:#440154;" />
        </svg>
        ${ranges[0]} - ${ranges[1]} <br>
        <!-- Color Box for Low-Medium Value -->
        <svg width="20" height="20">
        <rect width="20" height="20" style="fill:#31688e;" />
        </svg>
        ${ranges[1]} - ${ranges[2]} <br>
        <!-- Color Box for Medium Value -->
        <svg width="20" height="20">
        <rect width="20" height="20" style="fill:#35b779;" />
        </svg>
        ${ranges[2]} - ${ranges[3]} <br>
        <!-- Color Box for Medium-High Value -->
        <svg width="20" height="20">
        <rect width="20" height="20" style="fill:#8dd3c7;" />
        </svg>
        ${ranges[3]} - ${ranges[4]} <br>
        <!-- Color Box for High Value -->
        <svg width="20" height="20">
        <rect width="20" height="20" style="fill:#fde725;" />
        </svg>
        ${ranges[4]} - ${ranges[5]} <br>
        </div>
        '''

    m.get_root().html.add_child(folium.Element(legend_html))







##########
# Loop through the DataFrame to add each generator and line to supplier
for index, row in data.iterrows():
    # Generator location
    folium.Marker(
        [row['latitude'], row['longitude']],
        icon=folium.Icon(color='blue', icon='dot', icon_size=(24, 24)),
        popup=row['plant_name'],
    ).add_to(m)
    
    # Supplier location with popup showing the supplier's name
    folium.Marker(
        [row['current_supplier__hub_lat'], row['current_supplier__hub_lon']],
        # icon=folium.Icon(color='red', icon='briefcase'),
        icon=folium.Icon(color='red', icon='dot', icon_size=(28, 28)),
        popup=row['current_supplier__hub_name'],  # Use the supplier's name for the popup
    ).add_to(m)
    
    # Draw a line from generator to ACTUAL supplier
    folium.PolyLine(
        locations=[
            [row['latitude'], row['longitude']],
            [row['current_supplier__hub_lat'], row['current_supplier__hub_lon']]
        ],
        color='red', weight=4, opacity=0.5,
        popup=f"{round(row['percent_saving_by_choosing_current_supplier'],2)}% saving",  # Use the supplier's name for the popup
    ).add_to(m)

    # Draw a line from generator to CLOSEST supplier
    folium.PolyLine(
        locations=[
            [row['latitude'], row['longitude']],
            [row['closest_gas_hub_lat'], row['closest_gas_hub_lon']]
        ],
        color='black', weight=2, opacity=1
    ).add_to(m)

# Add a legend to the map
legend_html = '''
<div style="position: fixed; 
     top: 50px; right: 50px; width: 300px; height: 170px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding:10px; border-radius: 5px;">
     <b>Legend</b><br>
     <i style="background:black; width:50px; height:4px; display:inline-block;"></i> Connection to CLOSEST gas hub<br>
     <i style="background:red; width:50px; height:4px; display:inline-block;"></i> Connection to ACTUAL gas hub<br>
     <i style="background:blue; border-radius:50%; width:10px; height:10px; display:inline-block;"></i> Generator<br>
     <i style="background:red; border-radius:50%; width:10px; height:10px; display:inline-block;"></i> Gas hub<br>
     <i style="background:green; border-radius:50%; width:10px; height:10px; display:inline-block;"></i> "Orphan" Gas hub
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))


# Add an informational text box to the map
info_text_html = '''
<div style="position: fixed; 
     top: 220px; right: 50px; width: 600px; height: 200px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding:10px; border-radius: 5px; overflow-y: auto;">
     <b>Information</b><br>
     Red lines connect power generators with the gas hub they bought gas from.<br>
     Black lines connect power generators with the closest gas hub.<br>
     In cases where the generator bought gas from the closest gas hub, the line is red.<br>
     Click on the markers to see the name of the generator or gas hub.<br>
    The percentage saving is shown in the popup when clicking on the red lines.<br>
</div>
'''
m.get_root().html.add_child(folium.Element(info_text_html))


# Display the map
m.save(f'map_{distance}.html')

print(18)

