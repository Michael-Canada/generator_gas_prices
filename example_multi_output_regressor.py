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

# Impute NaN values in the target
imputer = SimpleImputer(strategy='mean')
generator_consumption_and_price_df_summary_pivot_imputed = imputer.fit_transform(generator_consumption_and_price_df_summary_pivot)

generator_consumption_and_price_df_summary_pivot_imputed = generator_consumption_and_price_df_summary_pivot_imputed[:,:4]

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

# Fit the model
regr = MultiOutputRegressor(Ridge(random_state=123, fit_intercept=True, positive=True, alpha=10.)).fit(hub_gas_prices_pivot, generator_consumption_and_price_df_summary_pivot_imputed)

# Define bounds for the intercept and coefficients
min_intercept = 0
max_intercept = 6

# Optimize the intercepts and coefficients using Lasso to enforce sparsity
for ind, estimator in enumerate(regr.estimators_):
    lasso = Lasso(alpha=1.0)  # You can adjust the alpha parameter to control the sparsity
    lasso.fit(hub_gas_prices_pivot, generator_consumption_and_price_df_summary_pivot_imputed[:, ind])
    estimator.intercept_ = lasso.intercept_
    estimator.coef_ = lasso.coef_.reshape(1, -1)

# Find the intercepts and coefficients
intercepts = [estimator.intercept_ for estimator in regr.estimators_]
coefficients = [estimator.coef_ for estimator in regr.estimators_]
print("Intercepts:", intercepts)
print("Coefficients:", coefficients)

# Predict
out = regr.predict(hub_gas_prices_pivot[[0]])
print(out)