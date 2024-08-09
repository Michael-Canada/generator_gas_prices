# environment: placebo_jupyter_env_new_python

import pandas as pd
import matplotlib.pyplot as plt
import folium
import numpy as np
import base64
from io import BytesIO

# the following creates the input for this script to run
# import example_multi_output_regressor_2


distance = 700
generators_with_close_hubs = pd.read_csv(f"generators_with_close_hubs_{distance}.csv")
generator_consumption_and_price_df_summary = pd.read_csv(f"generator_consumption_and_price_df_summary.csv")
hub_gas_prices_pivot = pd.read_pickle("hub_gas_prices_pivot.pkl")
predicted_current_gas_prices = pd.read_pickle("predicted_current_gas_prices.pkl")
NG_hub_loc_info = pd.read_parquet('/Users/michael.simantov/Documents/generator_gas_prices/ng_hub_definition_parquet.parquet')
generator_locations = pd.read_pickle("generator_locations.pkl")
coefficients = pd.read_pickle("coefficients.pkl")

NG_hub_coordinates = NG_hub_loc_info[['mv_symbol', 'latitude', 'longitude']].set_index('mv_symbol')


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


# Initialize the map at a central point
m = folium.Map(location=[37.0902, -95.7129], zoom_start=5)

# Add all gas hubs to the map
for index, row in NG_hub_loc_info.iterrows():
    # Create a marker for each hub
    folium.Marker(
        [row['latitude'], row['longitude']],  # Use the hub's latitude and longitude
        icon=folium.Icon(color='green', icon='square', icon_size=(24, 24)),  # Customize the icon
        popup=row['hub_name'],  # Use the hub's name for the popup
    ).add_to(m)  # Add the marker to the map

# Assuming 'data' now includes a 'current_supplier__hub_name' column
data = generators_with_close_hubs[['plant_id','plant_name','latitude', 'longitude', 'current_supplier__hub_lat', 'current_supplier__hub_lon', 'current_supplier__hub_name','current_supplier_hub_symbol', 'closest_gas_hub_lat', 'closest_gas_hub_lon','closest_gas_hub_symbol', 'percent_saving_by_choosing_current_supplier', 'quantity', 'current_supplier__Pvalue']]

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    
    return km

def find_similarity_between_hub_and_generator_prices(hub_gas_prices, generator_fuel_data_per_month):
    # merge the two dataframes
    fuel_prices = generator_fuel_data_per_month.merge(hub_gas_prices, on="month", how="inner").set_index('month')
    similarity = fuel_prices.values
    MSE = delta_prices = np.sqrt(   np.mean(   (  np.diff(similarity, axis=1)  )** 2   )   )
    mean_error = delta_prices = np.mean(np.diff(similarity, axis=1))
    delta_prices = np.diff(similarity, axis=1)
    
    return MSE, mean_error, delta_prices

# Loop through the DataFrame to add each generator and line to supplier
for index, row in data.iterrows():

    # generator_price_timeseries = generator_consumption_and_price_df_summary.loc[generator_consumption_and_price_df_summary['plant_id'] == row['plant_id']].sort_values(['year','month'])[['year','month','fuel_cost']]
    generator_price_timeseries = generator_consumption_and_price_df_summary.loc[generator_consumption_and_price_df_summary['plant_id'] == row['plant_id']].sort_values(['year','month'])[['month','fuel_cost']].set_index('month')
    CLOSEST_hub_gas_prices_timeseries = hub_gas_prices_pivot[row['closest_gas_hub_symbol']]
    # CURRENT_hub_gas_prices_timeseries = hub_gas_prices_pivot[row['current_supplier_hub_symbol']] # ZZZ
    OLD_hub_gas_prices_timeseries = hub_gas_prices_pivot[row['current_supplier_hub_symbol']] # ZZZ
    CURRENT_hub_gas_prices_timeseries = predicted_current_gas_prices[row['plant_name']]

# find_similarity_between_hub_and_generator_prices(hub_gas_prices, generator_price_timeseries)


    # Generate the plot
    fig, ax = plt.subplots()
    ax.plot(generator_price_timeseries)
    ax.plot(CLOSEST_hub_gas_prices_timeseries)
    ax.plot(OLD_hub_gas_prices_timeseries)
    ax.plot(CURRENT_hub_gas_prices_timeseries)
    plt.title(f'quantity: {int(row["quantity"])} MMBtu', fontsize=20)
    plt.legend(['Generator cost [$/MMBtu]', 'Closest Hub [$/MMBtu]', 'Old', 'Current Hub [$/MMBtu]'])

    # Save the plot to a PNG file in memory
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()


    # Create the HTML for the popup
    html = f"""
        <h4>{row['plant_name']}</h4>
        <img src="data:image/png;base64,{img_base64}" width="300" height="150", alt="Your Plot Title", style="border:0;", align="center", hspace="10", vspace="10", location=(110,0)>
    """
    iframe = folium.IFrame(html, width=350, height=250)
    popup = folium.Popup(iframe, max_width=350)


    # Generator location
    folium.Marker(
        [row['latitude'], row['longitude']],
        icon=folium.Icon(color='blue', icon='dot', icon_size=(24, 24)),
        popup=popup
    ).add_to(m)

    for coefs in coefficients[row['plant_name']].iteritems():
        if np.abs(coefs[1]) < .05:
            continue

        NG_hub_coord_lon = NG_hub_coordinates.loc[coefs[0], 'longitude']
        NG_hub_coord_lat = NG_hub_coordinates.loc[coefs[0], 'latitude']
        distance_to_hub = haversine_np(row['longitude'], row['latitude'], NG_hub_coord_lon, NG_hub_coord_lat)
        # Draw a line from generator to ACTUAL supplier
        folium.PolyLine(
            locations=[
                [row['latitude'], row['longitude']],
                [NG_hub_coord_lat, NG_hub_coord_lon]
            ],
            color='red', weight=4, opacity=(0.5 + coefs[1]/2),   #opacity is between 0.5 and 1 (1 being darkest)
            popup=f"coef: {round(coefs[1],2)}, distance: {int(distance_to_hub)}km",  # Use the supplier's name for the popup
        ).add_to(m)



        # Supplier location with popup showing the supplier's name
        folium.Marker(
            [NG_hub_coord_lat, NG_hub_coord_lon],
            # icon=folium.Icon(color='red', icon='briefcase'),
            icon=folium.Icon(color='red', icon='dot', icon_size=(28, 28)),
            popup=row['current_supplier__hub_name'],  # Use the supplier's name for the popup
        ).add_to(m)
    
    # # Draw a line from generator to ACTUAL supplier
    # folium.PolyLine(
    #     locations=[
    #         [row['latitude'], row['longitude']],
    #         [row['current_supplier__hub_lat'], row['current_supplier__hub_lon']]
    #     ],
    #     color='red', weight=4, opacity=0.5,
    #     popup=f"{round(row['percent_saving_by_choosing_current_supplier'],2)}% saving",  # Use the supplier's name for the popup
    # ).add_to(m)

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
     top: 220px; right: 50px; width: 600px; height: 400px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding:10px; border-radius: 5px; overflow-y: auto;">
     <b>Information</b><br>
     Red lines connect power generators with the gas hub they bought gas from.<br>
     Black lines connect power generators with the closest gas hub.<br>
     In cases where the generator bought gas from the closest gas hub, the black line hides the red line.<br><br>
     Click on the markers to see the name of the generator or gas hub.<br>
    The distance to the gas hub and its coefficient (out of 1) are shown in the popup when clicking on the red lines.<br><br>
    Click on a generator (blue dot) to see timeseries<br><br>
</div>
'''
m.get_root().html.add_child(folium.Element(info_text_html))


# # Display the map
# m.save(f'map_{distance}.html')

# print(18)

xxx = [1,2,3,4]
yyy = [11,22,33,22]
fig, ax = plt.subplots()
# ax.hist(xxx,yyy)
ax.hist(generators_with_close_hubs['percent_saving_by_choosing_current_supplier'],bins=15)
plt.title('% saving by choosing gas hubs', fontsize=16)
plt.xlabel('% saving on cost of gas compared to the default', fontsize=12)
plt.ylabel('Number of generators', fontsize=12)
plt.legend(['Generator cost [$/MMBtu]', 'Closest Hub [$/MMBtu]', 'Current Hub [$/MMBtu]'])


# Make the ticks on the axis larger
ax.tick_params(axis='both', which='major', labelsize=24)


# Adjust layout to ensure everything fits
plt.tight_layout()

#  Save the plot to a PNG file in memory
img = BytesIO()
plt.savefig(img, format='png')
img.seek(0)
img_base64 = base64.b64encode(img.getvalue()).decode()

# Create the HTML for the fixed-position image
# html = f"""
#     <div id="fixed-image" style="position:fixed; bottom:10px; left:10px; width:350px; height:400px; z-index:1000;">
#         <img src="data:image/png;base64,{img_base64}" width="450" height="350" alt="Plot Image">
#     </div>
# """
html = f"""
    <div id="fixed-image" style="position:fixed; bottom:10px; left:10px; width:450px; height:360px; z-index:1000;">
        <img src="data:image/png;base64,{img_base64}" width="450" height="350" alt="Plot Image">
    </div>
"""
m.get_root().html.add_child(folium.Element(html))

# Display the map
m.save(f'map_MS_{distance}.html')

print(18)

