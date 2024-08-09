# environment: placebo_jupyter_env_new_python

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


# Load data
hub_gas_prices_pivot = pd.read_pickle("hub_gas_prices_pivot.pkl")
generator_consumption_and_price_df_summary = pd.read_csv("generator_consumption_and_price_df_summary.csv")
generator_consumption_and_price_df_summary_pivot = generator_consumption_and_price_df_summary.pivot(index="month", columns="plant_name", values="fuel_cost")
NG_hub_loc_info = pd.read_parquet('/Users/michael.simantov/Documents/generator_gas_prices/ng_hub_definition_parquet.parquet')

generator_locations = generator_consumption_and_price_df_summary.groupby('plant_name').first()[['latitude', 'longitude']]
NG_mv_symbol_to_hub_name = NG_hub_loc_info.set_index('mv_symbol')['hub_name']

# Impute NaN values in the target
imputer = SimpleImputer(strategy='mean')
generator_consumption_and_price_df_summary_pivot_imputed = imputer.fit_transform(generator_consumption_and_price_df_summary_pivot)

# generator_consumption_and_price_df_summary_pivot_imputed = generator_consumption_and_price_df_summary_pivot_imputed[:,:5]

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    
    return km


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
        ridge_loss = np.mean((y - predictions) ** 2) + model.alpha * np.sum(model.coef_ ** 2)
        
        # Add a penalty for intercepts outside the bounds
        penalty = np.sum(np.maximum(0, intercept - max_intercept) ** 2) + np.sum(np.maximum(0, min_intercept - intercept) ** 2)
        
        # Add a penalty for coefficients not summing to 1
        coef_sum_penalty = (np.sum(coef) - 1) ** 2
        
        return ridge_loss + penalty + coef_sum_penalty

    # Fit the model
    regr = MultiOutputRegressor(Ridge(random_state=123, fit_intercept=True, positive=True, alpha=10.)).fit(hub_gas_prices_pivot, generator_consumption_and_price_df_summary_pivot_imputed)

    # Define bounds for the intercept and coefficients
    min_intercept = 0
    max_intercept = 6

    # Optimize the intercepts and coefficients
    for ind, estimator in enumerate(regr.estimators_):
        initial_params = np.concatenate(([estimator.intercept_], estimator.coef_.flatten()))
        bounds = [(min_intercept, max_intercept)] + [(0, None)] * len(estimator.coef_.flatten())
        result = minimize(custom_loss, initial_params, args=(hub_gas_prices_pivot, generator_consumption_and_price_df_summary_pivot_imputed[:, ind], estimator, min_intercept, max_intercept), bounds=bounds)
        estimator.intercept_ = result.x[0]
        estimator.coef_ = result.x[1:].reshape(1, -1)

else:

    def custom_cost_function(coefs, X, y, alpha, target_location):
        intercept = coefs[0]
        weights = coefs[1:]
        predictions = X.dot(weights) + intercept
        lasso_penalty = alpha * np.sum(np.abs(weights))
        mse = np.mean((y - predictions) ** 2)

        dist_to_hubs_penalty = 0
        for ind, w in enumerate(weights):
            gas_hub_mv_symbol = X.columns[ind]
            gas_hub_lon = NG_hub_loc_info.loc[NG_hub_loc_info['mv_symbol'] == gas_hub_mv_symbol, 'longitude'].values[0]
            gas_hub_lat = NG_hub_loc_info.loc[NG_hub_loc_info['mv_symbol'] == gas_hub_mv_symbol, 'latitude'].values[0]
            dist_to_hubs_penalty += haversine_np(target_location['longitude'], target_location['latitude'], gas_hub_lon, gas_hub_lat) * np.abs(w)
        
        dist_to_hubs_penalty /= 500

        return mse + lasso_penalty + dist_to_hubs_penalty

    # Fit the model
    regr = MultiOutputRegressor(Lasso(alpha=1.0)).fit(hub_gas_prices_pivot, generator_consumption_and_price_df_summary_pivot_imputed)
    # regr = MultiOutputRegressor(Lasso(alpha=1.0, positive=True)).fit(hub_gas_prices_pivot, generator_consumption_and_price_df_summary_pivot_imputed)

    # Define bounds for the intercept and coefficients
    min_intercept = 0
    max_intercept = 6

    intercepts = {}
    coefficients = {}
    predicted_current_gas_prices = {}
    # Optimize the intercepts and coefficients using custom cost function to enforce sparsity and constraints
    for ind, estimator in enumerate(regr.estimators_):
        name = generator_consumption_and_price_df_summary_pivot.columns[ind]
        X = hub_gas_prices_pivot
        y = generator_consumption_and_price_df_summary_pivot_imputed[:, ind]
        target_location = generator_locations.loc[name]
        alpha = 1.0  # You can adjust the alpha parameter to control the sparsity
        
        # Initial guess for the coefficients (intercept + weights)
        initial_coefs = np.concatenate(([estimator.intercept_], estimator.coef_))
        
        # Constraints: sum of weights = 1, intercept between 0 and 6, weights >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda coefs: np.sum(coefs[1:]) - 1},  # Sum of weights = 1
            {'type': 'ineq', 'fun': lambda coefs: coefs[0]},             # Intercept >= 0
            {'type': 'ineq', 'fun': lambda coefs: 6 - coefs[0]},         # Intercept <= 6
            {'type': 'ineq', 'fun': lambda coefs: coefs[1:]} #,             # Weights >= 0
            # {'type': 'ineq', 'fun': lambda coefs: 5 - np.sum(coefs[1:] > 0)}  # Max 3 non-zero weights
        ]
        
        # Minimize the custom cost function
        result = minimize(custom_cost_function, initial_coefs, args=(X, y, alpha, target_location), constraints=constraints)

        
        # Extract the optimized intercept and coefficients
        optimized_intercept = np.round(result.x[0], 2)
        optimized_coefs = np.round(result.x[1:], 2)

        print(f"{ind}: optimized_intercept")
        print([c for c in optimized_coefs if np.abs(c) > 0.01])
        
        intercepts[name] = optimized_intercept
        coefficients[name] = optimized_coefs

        prediction = hub_gas_prices_pivot.dot(optimized_coefs) + optimized_intercept
        predicted_current_gas_prices[name] = prediction.values
        if False:
            plt.plot(prediction.values, label='prediction')
            plt.plot(y, label='target')
            plt.legend()
            plt.close()


#convert the dict predicted_current_gas_prices to a DataFrame. The index of the DataFrame is the index of the hub_gas_prices_pivot
predicted_current_gas_prices = pd.DataFrame(predicted_current_gas_prices, index=hub_gas_prices_pivot.index)
coefficients_NG_hub_name = pd.DataFrame(coefficients, index=NG_mv_symbol_to_hub_name[hub_gas_prices_pivot.columns])
coefficients = pd.DataFrame(coefficients, index=hub_gas_prices_pivot.columns)
intercepts = pd.DataFrame.from_dict(intercepts, orient='index', columns=['Value'])

# save the dict predicted_current_gas_prices
predicted_current_gas_prices.to_pickle("predicted_current_gas_prices.pkl")
generator_locations.to_pickle("generator_locations.pkl")
coefficients.to_pickle("coefficients.pkl")
coefficients_NG_hub_name.to_pickle("coefficients_NG_hub_name.pkl")
intercepts.to_pickle("intercepts.pkl")

# Find the intercepts and coefficients

print("Intercepts:", intercepts)
# print("Coefficients:", coefficients)


print(18)
