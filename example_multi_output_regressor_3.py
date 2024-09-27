# environment: placebo_jupyter_env_new_python

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# generator_consumption_and_price_df_summary[generator_consumption_and_price_df_summary['plant_name'].str.contains('Sabine')]
# Load data
hub_gas_prices_pivot = pd.read_pickle("hub_gas_prices_pivot.pkl")


# get rid of columns in hub_gas_prices_pivot if there are NaN values in the columns
columns_without_nan = hub_gas_prices_pivot.columns[~hub_gas_prices_pivot.isnull().any()]
columns_with_nan = hub_gas_prices_pivot.columns[hub_gas_prices_pivot.isnull().any()]
hub_gas_prices_pivot.drop(columns=["#NIETGZ1D"], inplace=True)
# Impute NaN values in the gas prices
imputer = SimpleImputer(strategy="mean")
hub_gas_prices_pivot_imputed = imputer.fit_transform(hub_gas_prices_pivot)

# Convert the result back to a DataFrame
hub_gas_prices_pivot = pd.DataFrame(
    hub_gas_prices_pivot_imputed,
    columns=hub_gas_prices_pivot.columns,
    index=hub_gas_prices_pivot.index,
)

generator_consumption_and_price_df_summary = pd.read_csv(
    "generator_consumption_and_price_df_summary.csv"
)

generator_consumption_and_price_df_summary_pivot = (
    generator_consumption_and_price_df_summary.pivot(
        index="year_month", columns="plant_name", values="fuel_cost"
    )
)


# creater a dictionary from plant_name to plant_id:
plant_name_to_id = (
    generator_consumption_and_price_df_summary.groupby("plant_name")["plant_id"]
    .first()
    .to_dict()
)

# Group by 'month' and 'plant_id', then sum the 'quantity'
plant_month_quantity_df = (
    generator_consumption_and_price_df_summary.groupby(["year_month", "plant_id"])[
        "quantity"
    ]
    .sum()
    .reset_index()
)


# Find 'quantity' per month for the plant with plant_id
def find_quantities_per_plant_id(plant_id):
    ans = plant_month_quantity_df[
        plant_month_quantity_df["plant_id"] == plant_id
    ].sort_values("year_month")

    # fill in for missing months with 0
    # all_months = pd.DataFrame({"year_month": range(1, 13)})
    # ans = pd.merge(all_months, ans, on="year_month", how="left")
    # ans["quantity"].fillna(0, inplace=True)
    date_range = (
        pd.date_range(start="2022-01", end="2024-06", freq="MS")
        .strftime("%Y-%m")
        .tolist()
    )
    all_months = pd.DataFrame({"year_month": date_range})
    ans = pd.merge(all_months, ans, on="year_month", how="left")
    ans["quantity"].fillna(0, inplace=True)

    ans["plant_id"] = plant_id

    return ans


NG_hub_loc_info = pd.read_parquet(
    "/Users/michael.simantov/Documents/generator_gas_prices/ng_hub_definition_parquet.parquet"
)

generator_locations = generator_consumption_and_price_df_summary.groupby(
    "plant_name"
).first()[["latitude", "longitude"]]
NG_mv_symbol_to_hub_name = NG_hub_loc_info.set_index("mv_symbol")["hub_name"]

# Impute NaN values in the target
imputer = SimpleImputer(strategy="mean")
generator_consumption_and_price_df_summary_pivot_imputed = imputer.fit_transform(
    generator_consumption_and_price_df_summary_pivot
)

generator_consumption_and_price_df_summary_pivot = (
    generator_consumption_and_price_df_summary_pivot[
        generator_consumption_and_price_df_summary_pivot.index.isin(
            hub_gas_prices_pivot.index
        )
    ]
)
hub_gas_prices_pivot = hub_gas_prices_pivot[
    hub_gas_prices_pivot.index.isin(
        generator_consumption_and_price_df_summary_pivot.index
    )
]


# generator_consumption_and_price_df_summary_pivot_imputed = generator_consumption_and_price_df_summary_pivot_imputed[:,:5]


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c

    return km


combined_df = pd.DataFrame(
    columns=["year_month", "actual_price", "predicted_price", "unit_id", "Quantity"]
)


# Define custom loss function
if False:

    def custom_loss(params, X, y, model, min_intercept, max_intercept):
        intercept = params[0]
        coef = params[1:]

        # Set the intercept and coefficients of the model
        model.intercept_ = intercept
        model.coef_ = coef.reshape(1, -1)

        # Predict using the model
        predictions = model.predict(X)

        # Calculate the Ridge regression loss
        ridge_loss = np.mean((y - predictions) ** 2) + model.alpha * np.sum(
            model.coef_**2
        )

        # Add a penalty for intercepts outside the bounds
        penalty = np.sum(np.maximum(0, intercept - max_intercept) ** 2) + np.sum(
            np.maximum(0, min_intercept - intercept) ** 2
        )

        # Add a penalty for coefficients not summing to 1
        coef_sum_penalty = (np.sum(coef) - 1) ** 2

        return ridge_loss + penalty + coef_sum_penalty

    # Fit the model
    regr = MultiOutputRegressor(
        Ridge(random_state=123, fit_intercept=True, positive=True, alpha=10.0)
    ).fit(
        hub_gas_prices_pivot, generator_consumption_and_price_df_summary_pivot_imputed
    )

    # Define bounds for the intercept and coefficients
    min_intercept = 0
    max_intercept = 6

    # Optimize the intercepts and coefficients
    for ind, estimator in enumerate(regr.estimators_):
        initial_params = np.concatenate(
            ([estimator.intercept_], estimator.coef_.flatten())
        )
        bounds = [(min_intercept, max_intercept)] + [(0, None)] * len(
            estimator.coef_.flatten()
        )
        result = minimize(
            custom_loss,
            initial_params,
            args=(
                hub_gas_prices_pivot,
                generator_consumption_and_price_df_summary_pivot_imputed[:, ind],
                estimator,
                min_intercept,
                max_intercept,
            ),
            bounds=bounds,
        )
        estimator.intercept_ = result.x[0]
        estimator.coef_ = result.x[1:].reshape(1, -1)

else:

    def custom_cost_function(coefs, X, y, alpha, target_location, Quantities):
        intercept = coefs[0]
        weights = coefs[1:]
        predictions = X.dot(weights) + intercept
        lasso_penalty = alpha * np.sum(np.abs(weights))

        mse_weights = 1 / ((y - np.median(y)) * (y - np.median(y)) + 1e-6)
        # mse_weights = mse_weights / np.sum(mse_weights)
        mse_weights = Quantities.quantity.values.copy()
        mse_weights /= np.sum(mse_weights)
        mse = np.mean(((y - predictions) * mse_weights) ** 2) * 10000

        dist_to_hubs_penalty = 0
        for ind, w in enumerate(weights):
            gas_hub_mv_symbol = X.columns[ind]
            gas_hub_lon = NG_hub_loc_info.loc[
                NG_hub_loc_info["mv_symbol"] == gas_hub_mv_symbol, "longitude"
            ].values[0]
            gas_hub_lat = NG_hub_loc_info.loc[
                NG_hub_loc_info["mv_symbol"] == gas_hub_mv_symbol, "latitude"
            ].values[0]
            dist_to_hubs_penalty += haversine_np(
                target_location["longitude"],
                target_location["latitude"],
                gas_hub_lon,
                gas_hub_lat,
            ) * np.abs(w)

        dist_to_hubs_penalty /= 500

        return mse + lasso_penalty + dist_to_hubs_penalty

    # Fit the model
    regr = MultiOutputRegressor(Lasso(alpha=1.0)).fit(
        hub_gas_prices_pivot, generator_consumption_and_price_df_summary_pivot_imputed
    )
    # regr = MultiOutputRegressor(Lasso(alpha=1.0, positive=True)).fit(hub_gas_prices_pivot, generator_consumption_and_price_df_summary_pivot_imputed)

    # Define bounds for the intercept and coefficients
    min_intercept = 0
    max_intercept = 4  # zzz

    intercepts = {}
    coefficients = {}
    predicted_current_gas_prices = {}
    # Optimize the intercepts and coefficients using custom cost function to enforce sparsity and constraints
    for ind, estimator in enumerate(regr.estimators_):
        name = generator_consumption_and_price_df_summary_pivot.columns[ind]
        unit_id = plant_name_to_id[name]
        Quantities = find_quantities_per_plant_id(unit_id)
        # if not name in ["Port Washington Generating Station", "Sabine"]:
        #     continue
        # if not name == "Sabine":
        #     continue
        # if not unit_id == 6137:
        #     continue
        X = hub_gas_prices_pivot
        y = generator_consumption_and_price_df_summary_pivot_imputed[:, ind]
        target_location = generator_locations.loc[name]
        alpha = 1.0  # You can adjust the alpha parameter to control the sparsity

        # Initial guess for the coefficients (intercept + weights)
        initial_coefs = np.concatenate(([estimator.intercept_], estimator.coef_))

        # Constraints: sum of weights = 1, intercept between 0 and max_intercept, weights >= 0
        constraints = [
            {
                "type": "eq",
                "fun": lambda coefs: np.sum(coefs[1:]) - 1,
            },  # Sum of weights = 1
            {"type": "ineq", "fun": lambda coefs: coefs[0]},  # Intercept >= 0
            {
                "type": "ineq",
                "fun": lambda coefs: max_intercept - coefs[0],
            },  # Intercept <= max_intercept
            {
                "type": "ineq",
                "fun": lambda coefs: coefs[1:],
            },  # ,             # Weights >= 0
            # {'type': 'ineq', 'fun': lambda coefs: 5 - np.sum(coefs[1:] > 0)}  # Max 3 non-zero weights
        ]

        # Minimize the custom cost function
        result = minimize(
            custom_cost_function,
            initial_coefs,
            args=(X, y, alpha, target_location, Quantities),
            constraints=constraints,
        )

        # Extract the optimized intercept and coefficients
        optimized_intercept = np.round(result.x[0], 2)
        optimized_coefs = np.round(result.x[1:], 2)

        print(f"{ind}  optimized_intercepts: plant_name: {name}")
        print([c for c in optimized_coefs if np.abs(c) > 0.01])

        intercepts[name] = optimized_intercept
        coefficients[name] = optimized_coefs

        prediction = hub_gas_prices_pivot.dot(optimized_coefs) + optimized_intercept
        predicted_current_gas_prices[name] = prediction.values
        if False:
            plt.plot(prediction.values, label="prediction")
            plt.plot(y, label="target")
            plt.legend()
            plt.close()

        # keep the findings in a DataFrame
        data = {
            # "month": list(range(1, 19)),
            # "unit_id": [unit_id] * 18,
            "year_month": X.index.tolist(),
            "actual_price": y,
            "predicted_price": prediction,
        }

        df = pd.DataFrame(data)
        df["unit_id"] = unit_id
        df["Quantity"] = Quantities.quantity.values
        combined_df = pd.concat([combined_df, df], ignore_index=True)

        # print(18)


# convert the dict predicted_current_gas_prices to a DataFrame. The index of the DataFrame is the index of the hub_gas_prices_pivot
predicted_current_gas_prices = pd.DataFrame(
    predicted_current_gas_prices, index=hub_gas_prices_pivot.index
)
coefficients_NG_hub_name = pd.DataFrame(
    coefficients, index=NG_mv_symbol_to_hub_name[hub_gas_prices_pivot.columns]
)
coefficients = pd.DataFrame(coefficients, index=hub_gas_prices_pivot.columns)
intercepts = pd.DataFrame.from_dict(intercepts, orient="index", columns=["Value"])

# save the dict predicted_current_gas_prices
# combined_df.to_csv("combined_df_OLD.csv")
# predicted_current_gas_prices.to_pickle("predicted_current_gas_prices_OLD.pkl")
# generator_locations.to_pickle("generator_locations_OLD.pkl")
# coefficients.to_pickle("coefficients_OLD.pkl")
# coefficients_NG_hub_name.to_pickle("coefficients_NG_hub_name_OLD.pkl")
# intercepts.to_pickle("intercepts_OLD.pkl")


combined_df.to_csv("combined_df.csv")
predicted_current_gas_prices.to_pickle("predicted_current_gas_prices.pkl")
generator_locations.to_pickle("generator_locations.pkl")
coefficients.to_pickle("coefficients.pkl")
coefficients_NG_hub_name.to_pickle("coefficients_NG_hub_name.pkl")
intercepts.to_pickle("intercepts.pkl")

# Find the intercepts and coefficients

print("Intercepts:", intercepts)
# print("Coefficients:", coefficients)


print(18)
