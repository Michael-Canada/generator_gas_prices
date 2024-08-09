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


# # Display the map
# m.save(f'map_{distance}.html')

# print(18)




# Display the map
m.save(f'map_{distance}.html')

print(18)

