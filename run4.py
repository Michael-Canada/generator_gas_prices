# environment: placebo_jupyter_env_new_python

from io import BytesIO
from urllib.parse import urlparse

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import NamedTuple
import pytz
from datetime import date
import io
import os

plt.rcParams["figure.figsize"] = (10, 7)
import requests
import duckdb
import json
import jsonplus
import pickle
import numpy as np
import scipy.stats as stats

import input_reader

GO_TO_GCLOUD = True

# ISO_name = "MISO"
ISO_name = "SWPP"

# CLOSEST_INTERESTING_GAS_HUB = 1600

# Example of a parquet file:
# file = pd.read_parquet('/Users/michael.simantov/Documents/generator_gas_prices/2024-07-16.parquet')

# dt_str = dt.strftime("%Y%m%d")

# df_cases = _get_dfm("https://api1.marginalunit.com/reflow/ercot-rt-se/cases?columns=code,case_timestamp")
# df_cases.case_timestamp = pd.to_datetime(df_cases.case_timestamp, utc=True).dt.tz_convert("US/Central")
# df_cases = df_cases.set_index("code")


# this function gets the coordinates (lat and lon) of two hubs, and finds the geographic distance in km between them, based on trigonomatric calculations:
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c

    return km


# distane_between_hubs = _geo_distance(35.694, 51.42, 32.08, 34.7)
def _geo_distance(vec_1_lon, vec_1_lat, vec_2_lon, vec_2_lat):
    geo_dist = haversine_np(vec_1_lon, vec_1_lat, vec_2_lon, vec_2_lat)

    return geo_dist


# # 1) Geo distance
#     df_psse_c["geo_distance"] = df_psse_c[["station", "latitude", "longitude"]].apply(
#         lambda r: _geo_distance(r.values, eac_row.values), axis=1
#     )


def get_hub_gas_prices():
    dt_str = "2023-"
    benchmark = duckdb.read_parquet(
        "/Users/michael.simantov/Documents/generator_gas_prices/*.parquet",
        filename=True,
    )
    hub_prices = duckdb.sql(
        f"SELECT * FROM benchmark WHERE filename LIKE '%{dt_str}%'"
    ).to_df()
    hub_prices["month"] = hub_prices.filename.map(
        lambda f: int(f.split("/")[-1].split(".")[0].split("-")[1])
    )
    # hub_prices["timestamp"] = hub_prices.case.map(df_cases.case_timestamp)

    hub_prices = hub_prices.drop(
        columns=[
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
            "published_timestamp",
            "parent_dataset",
            "parent_environment",
            "parent_location",
            "parent_version",
        ]
    )
    # hub_prices = hub_prices.set_index("month")

    return hub_prices


# the function reads data from an excel document:
def read_data(file):

    with open(file, "rb") as fdesc:
        # data = input_reader.read_generation_and_fuel_data(fdesc)
        data = input_reader.read_fuel_receipts_and_costs(fdesc)

    return data


def weighted_average(group):
    weighted_sum = (group["quantity"] * group["fuel_cost"]).sum()
    total_quantity = group["quantity"].sum()
    if total_quantity > 0:
        return weighted_sum / total_quantity
    else:
        return 0


def _get_auth(env_var: str = "SELF"):
    return tuple(os.environ[env_var].split(":"))


AUTH = _get_auth()


def _get_dfm(url, auth=AUTH):
    resp = requests.get(url, auth=auth)

    if resp.status_code != 200:
        print(resp.text)
        resp.raise_for_status()

    dfm = pd.read_csv(io.StringIO(resp.text))
    return dfm


def find_closest_hubs(
    generator_consumption_and_price_df_summary,
    NG_hub_loc_info,
    hub_gas_prices_pivot,
    CLOSEST_INTERESTING_GAS_HUB,
):

    def find_similarity_between_hub_and_generator_prices(
        hub_gas_prices, generator_fuel_data_per_month
    ):
        # merge the two dataframes
        fuel_prices = generator_fuel_data_per_month.merge(
            hub_gas_prices, on="month", how="inner"
        ).set_index("month")
        similarity = fuel_prices.values
        MSE = np.sqrt(np.mean((np.diff(similarity, axis=1)) ** 2))
        mean_error = np.mean(np.diff(similarity, axis=1))
        delta_prices = np.diff(similarity, axis=1)

        hub_more_expensive_than_generator = np.diff(similarity, axis=1) > 0
        hub_cheaper_than_generator = np.diff(similarity, axis=1) < 0

        DONT_USE_THIS_HUB = False
        if (
            np.sum(hub_more_expensive_than_generator)
            > np.sum(hub_cheaper_than_generator)
            and mean_error > 0
        ):
            DONT_USE_THIS_HUB = True

        return MSE, mean_error, delta_prices, DONT_USE_THIS_HUB

    generators = (
        generator_consumption_and_price_df_summary.groupby("plant_id")
        .first()[["quantity", "fuel_cost", "plant_name", "latitude", "longitude"]]
        .reset_index()
    )
    # create a list of distances between each generator and each hub

    for i in range(len(generators)):
        distances = []
        generator = generators.iloc[i]
        # if not generator['plant_name'] == 'Attala':
        #     continue

        for max_allowed_distance in range(CLOSEST_INTERESTING_GAS_HUB, 10000, 100):

            for j in range(len(NG_hub_loc_info)):
                hub = NG_hub_loc_info.iloc[j]
                distance = _geo_distance(
                    generator["longitude"],
                    generator["latitude"],
                    hub["longitude"],
                    hub["latitude"],
                )
                if distance > max_allowed_distance:
                    continue

                generator_fuel_data_per_month = (
                    generator_consumption_and_price_df_summary[
                        generator_consumption_and_price_df_summary.plant_id
                        == generator.plant_id
                    ].sort_values(["year", "month"])[["month", "fuel_cost"]]
                )  # ['fuel_cost'].values
                if len(generator_fuel_data_per_month) < 3:
                    continue
                hub_gas_prices = hub_gas_prices_pivot[hub["mv_symbol"]].reset_index(
                    "month"
                )
                MSE, mean_error, delta_prices, DONT_USE_THIS_HUB = (
                    find_similarity_between_hub_and_generator_prices(
                        hub_gas_prices, generator_fuel_data_per_month
                    )
                )
                if (
                    DONT_USE_THIS_HUB
                ):  # True means that the gas hub shows prices higher than what were actually paid by the generator
                    continue
                # _, p_value_matching_generator_hub = stats.ttest_1samp(delta_prices, 0)   #check Normal around 0. Not good if there is a bias
                _, p_value_matching_generator_hub = stats.shapiro(
                    delta_prices
                )  # check Normal, not necessarily around 0. Good if there is a bias

                # mean_error is hub_price - generator_paid_price
                mean_hub_price = np.mean(hub_gas_prices.set_index("month").values)

                if p_value_matching_generator_hub > 0:  # 0.05:
                    distances.append(
                        (
                            j,
                            generator["plant_id"],
                            distance,
                            float(p_value_matching_generator_hub),
                            mean_error,
                            hub["mv_symbol"],
                            hub["hub_name"],
                            hub["latitude"],
                            hub["longitude"],
                            mean_hub_price,
                        )
                    )

            if len(distances) > 0:
                break

        # How to find the hub that the current generator bought from:
        # If more than one hub has p_value > 0.5 then choose the one that is closest to the generator
        # else: From the 5 hubs with the highest p_value, choose the one with the lowest gas price
        # note that p_value in the two lines above mean how well each hub's gas prices match the generator's paid prices
        distances_df = pd.DataFrame(
            distances,
            columns=[
                "index",
                "plant_id",
                "distance",
                "p_value",
                "mean_error",
                "hub_symbol",
                "hub_name",
                "latitude",
                "longitude",
                "mean_hub_price",
            ],
        ).reset_index()
        best_matches_for_observed_generator_pay = distances_df[
            distances_df["p_value"] >= 0.5
        ]
        if len(best_matches_for_observed_generator_pay) == 0:
            # find the hub with highest p-value:
            distances_to_paid_hub = sorted(distances, key=lambda x: x[3])[
                -5:
            ]  # the 3 most similar hubs to what the generator paid

            # being here means we did not find a hub with gas prices similar enough to what the generator paid.
            # We will choose the hub that is the cheapest one, under the assumption that that's what the generator operators do
            distances_to_paid_hub = sorted(distances_to_paid_hub, key=lambda x: x[4])[
                0
            ]  # the 3 cheapest hubs   zzz

        else:
            # chosen_ind = best_matches_for_observed_generator_pay.sort_values('mean_error')['index'].values[0]
            chosen_ind = best_matches_for_observed_generator_pay.sort_values(
                "distance"
            )["index"].values[0]
            distances_to_paid_hub = [f for f in distances if f[0] == chosen_ind][0]

        # find the hub with gas prices closest to what was paid:
        distances_to_closest_hub = sorted(distances, key=lambda x: x[2])[0]

        generators.loc[i, "closest_gas_hub_symbol"] = distances_to_closest_hub[5]
        generators.loc[i, "distance_to_closest_gas_hub"] = distances_to_closest_hub[2]
        generators.loc[i, "closest_gas_hub_name"] = distances_to_closest_hub[6]
        generators.loc[i, "closest_gas_hub_lat"] = distances_to_closest_hub[7]
        generators.loc[i, "closest_gas_hub_lon"] = distances_to_closest_hub[8]

        generators.loc[i, "current_supplier_hub_symbol"] = distances_to_paid_hub[5]
        generators.loc[i, "distance_to_current_supplier"] = distances_to_paid_hub[2]
        generators.loc[i, "current_supplier__hub_name"] = distances_to_paid_hub[6]
        generators.loc[i, "current_supplier__hub_lat"] = distances_to_paid_hub[7]
        generators.loc[i, "current_supplier__hub_lon"] = distances_to_paid_hub[8]
        generators.loc[i, "current_supplier__Pvalue"] = distances_to_paid_hub[3]

        percent_saving_by_choosing_current_supplier = int(
            100
            * (distances_to_closest_hub[4] - distances_to_paid_hub[4])
            / generator_fuel_data_per_month.fuel_cost.mean()
        )
        dollar_saving_by_choosing_current_supplier = round(
            distances_to_closest_hub[4] - distances_to_paid_hub[4], 2
        )

        generators.loc[i, "dollar_saving_by_choosing_current_supplier"] = round(
            dollar_saving_by_choosing_current_supplier, 2
        )
        generators.loc[i, "percent_saving_by_choosing_current_supplier"] = (
            percent_saving_by_choosing_current_supplier
        )

    return generators


def get_generator_coordinates_and_name():
    # in order to save cost of using the google cloud, we will download the data from the API and save it in a csv file
    if GO_TO_GCLOUD:
        df_eia_miso = _get_dfm(
            "https://api1.marginalunit.com/misc-data/eia/generators/monthly?columns=plant_id,plant_name,latitude,longitude"
        )
        df_eia_miso.to_csv("df_eia_miso.csv", index=False)
    else:
        df_eia_miso = pd.read_csv("df_eia_miso.csv")

    df_eia_miso = (
        df_eia_miso.groupby(["plant_id", "plant_name"])
        .agg({"latitude": "first", "longitude": "first"})
        .reset_index()
    )

    return df_eia_miso


NG_hub_loc_info = pd.read_parquet(
    "/Users/michael.simantov/Documents/generator_gas_prices/ng_hub_definition_parquet.parquet"
)
hub_gas_prices = get_hub_gas_prices()

hub_gas_prices = hub_gas_prices[
    hub_gas_prices.price_symbol.isin(NG_hub_loc_info.mv_symbol)
]
NG_hub_loc_info = NG_hub_loc_info[
    NG_hub_loc_info.mv_symbol.isin(hub_gas_prices.price_symbol)
]
NG_hub_loc_info.to_csv(f"NG_hub_loc_info.csv", index=False)

hub_gas_prices_per_month = (
    hub_gas_prices.groupby(["month", "price_symbol"])
    .agg({"mid_point": "mean"})
    .reset_index()
)
hub_gas_prices_pivot = hub_gas_prices_per_month.pivot(
    index="month", columns="price_symbol", values="mid_point"
)
hub_gas_prices_pivot.sort_values("month", ascending=True)  # ZZZ
# hub_gas_prices_pivot["mean_mid_point"] = hub_gas_prices_pivot.mean(axis=1)

# the function reads data from an excel document:
file = "EIA923_Schedules_2_3_4_5_M_12_2023_Early_Release.xlsx"
data = read_data(file)

generator_consumption_and_price = []
this_data = []
for d in data:
    if d.fuel_group == "Natural Gas" and d.ba_code == ISO_name:
        this_data = [
            d.year,
            d.month,
            d.plant_id,
            d.plant_name,
            d.quantity,
            d.average_heat_content,
            d.fuel_cost,
            d.fuel_group,
            d.ba_code,
        ]
        generator_consumption_and_price.append(this_data)

generator_consumption_and_price_df = pd.DataFrame(
    generator_consumption_and_price,
    columns=[
        "year",
        "month",
        "plant_id",
        "plant_name",
        "quantity",
        "average_heat_content",
        "fuel_cost",
        "fuel_group",
        "balancing_authority_code",
    ],
)
# generator_consumption_and_price_df['fuel_cost'] = generator_consumption_and_price_df['fuel_cost'] / generator_consumption_and_price_df['average_heat_content']
generator_consumption_and_price_df.drop(columns=["average_heat_content"], inplace=True)


generator_consumption_and_price_df = generator_consumption_and_price_df.dropna()  # ZZZ

# calculate the weighted average of the fuel cost for each plant
generator_consumption_and_price_df_summary = (
    generator_consumption_and_price_df.groupby(["year", "month", "plant_id"])
    .apply(
        lambda x: pd.Series(
            {
                "quantity": x["quantity"].sum(),
                "fuel_cost": weighted_average(x) / 100,  # cents to dollars
                "number_of_suppliers": int(len(x)),
            }
        )
    )
    .reset_index()
)

# get generators' coordinates and name
generator_more_data = get_generator_coordinates_and_name()

# merge the two dataframes
generator_consumption_and_price_df_summary = (
    generator_consumption_and_price_df_summary.merge(generator_more_data, on="plant_id")
)
generator_consumption_and_price_df_summary.to_csv(
    f"generator_consumption_and_price_df_summary.csv", index=False
)
hub_gas_prices_pivot.to_pickle("hub_gas_prices_pivot.pkl")

############### TESTING
if True:
    results = []
    for ff in range(200, 2000, 100):
        generators_with_close_hubs = find_closest_hubs(
            generator_consumption_and_price_df_summary,
            NG_hub_loc_info,
            hub_gas_prices_pivot,
            ff,
        )
        mm = (
            generators_with_close_hubs.percent_saving_by_choosing_current_supplier.mean()
        )
        ss = (
            generators_with_close_hubs.percent_saving_by_choosing_current_supplier.std()
        )
        pp = round(mm / (ss / np.sqrt(108)), 1)
        results.append((ff, mm, ss, pp))

        generators_with_close_hubs.to_csv(
            f"generators_with_close_hubs_{ff}.csv", index=False
        )

    # convert 'results' into a dataframe:
    results_df = pd.DataFrame(
        results, columns=["max-distance", "mean", "std", "confidence"]
    )
    # plot results_df['confidence'] vs. results_df['max-distance']
    plt.plot(results_df["max-distance"], results_df["confidence"])

    plt.figure()
    plt.plot(results_df["max-distance"], results_df["mean"])


###############################

# find 3 closest gas hubs to each generator
plt.figure()
generators_with_close_hubs = find_closest_hubs(
    generator_consumption_and_price_df_summary,
    NG_hub_loc_info,
    hub_gas_prices_pivot,
    700,
)

# save results to a csv file:
generators_with_close_hubs.to_csv("generators_with_close_hubs.csv", index=False)

generators_with_close_hubs.percent_saving_by_choosing_current_supplier.hist()
plt.title(
    "Histogram of percent saving on gas supply by choosing current supplier over the closest hub",
    fontsize=18,
)
plt.xlabel("Percent saving", fontsize=18)
plt.ylabel("Number of occurences", fontsize=18)
plt.axis([-40, 40, 0, 25])

# plt.figure()
# generators_with_close_hubs[['distance_to_closest_gas_hub', 'distance_to_current_supplier']].plot()

# plt.figure()
# generators_with_close_hubs['current_supplier__Pvalue'].plot()

# Create a figure and a set of subplots
fig, axs = plt.subplots(
    3, 1, figsize=(10, 8), sharex=True
)  # 2 rows, 1 column, sharing x-axis
generators_with_close_hubs[
    ["distance_to_closest_gas_hub", "distance_to_current_supplier"]
].plot(ax=axs[0])
axs[0].set_title("Distance to Closest Gas Hub and Current Supplier")
generators_with_close_hubs["current_supplier__Pvalue"].plot(ax=axs[1])
axs[1].set_title("Current Supplier P-value")
generators_with_close_hubs["percent_saving_by_choosing_current_supplier"].plot(
    ax=axs[2]
)
# plt.tight_layout()

##########

# plt.figure()
# find 3 closest gas hubs to each generator
generators_with_close_hubs = find_closest_hubs(
    generator_consumption_and_price_df_summary,
    NG_hub_loc_info,
    hub_gas_prices_pivot,
    700,
)

# Create a figure and a set of subplots
# fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column, sharing x-axis
generators_with_close_hubs[
    ["distance_to_closest_gas_hub", "distance_to_current_supplier"]
].plot(ax=axs[0])
axs[0].set_title("Distance to Closest Gas Hub and Current Supplier")
generators_with_close_hubs["current_supplier__Pvalue"].plot(ax=axs[1])
axs[1].set_title("Current Supplier P-value")
generators_with_close_hubs["percent_saving_by_choosing_current_supplier"].plot(
    ax=axs[2]
)
axs[2].set_title("Percent savings by choosing current supplier over closest hub")
plt.tight_layout()

if False:
    plt.figure()
    generators_with_close_hubs.percent_saving_by_choosing_current_supplier.hist()
    plt.title(
        "Histogram of percent saving on gas supply by choosing current supplier over the closest hub",
        fontsize=18,
    )
    plt.xlabel("Percent saving", fontsize=18)
    plt.ylabel("Number of occurences", fontsize=18)
    plt.axis([-40, 40, 0, 25])


# plt.show()


if False:

    # find correlation between the mid_point of all hubs
    correlation_between_hubs = hub_gas_prices_pivot.corr()

    # create a table of distances between the hubs, whose coordinates are given in the NG_hub_loc_info table (columns: latitude, longitude)
    def calculate_distances(hubs_info):
        distances = []
        for i in range(len(hubs_info)):
            for j in range(len(hubs_info)):
                hub1 = hubs_info.iloc[i]
                hub2 = hubs_info.iloc[j]
                distance = _geo_distance(
                    hub1["latitude"],
                    hub1["longitude"],
                    hub2["latitude"],
                    hub2["longitude"],
                )
                distances.append(((hub1["hub_name"], hub2["hub_name"]), distance))
        return distances

    # Calculate distances and create a DataFrame
    distances = calculate_distances(NG_hub_loc_info)
    distances_df = pd.DataFrame(distances, columns=["Hub Pair", "Distance"])
    distances_df[["Hub1", "Hub2"]] = pd.DataFrame(
        distances_df["Hub Pair"].tolist(), index=distances_df.index
    )
    distances_df.drop(columns=["Hub Pair"], inplace=True)

    # create a pivot table of the distances between the hubs
    distances_pivot = distances_df.pivot(
        index="Hub1", columns="Hub2", values="Distance"
    )

    # price_diff_between_hubs = hub_gas_prices_pivot.diff(axis=1)

    hub_gas_prices = hub_gas_prices.groupby(["month", "price_symbol"]).agg(
        {"mid_point": "mean"}
    )

    print(18)

# # merge 'generators_with_close_hubs' and 'generator_consumption_and_price_df_summary'
# merged = pd.merge(generators_with_close_hubs, generator_consumption_and_price_df_summary, on='plant_id')
# merged2 = merged[['number_of_suppliers','current_supplier__Pvalue']].sort_values('number_of_suppliers').reset_index()
# merged2.plot(x='number_of_suppliers', y='current_supplier__Pvalue')


# print(18)
