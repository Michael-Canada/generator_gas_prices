import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.impute import SimpleImputer


if True:
    from sklearn.datasets import load_linnerud
    X, y = load_linnerud(return_X_y=True)

# Impute NaN values in the target
    imputer = SimpleImputer(strategy='mean')
    y_imputed = imputer.fit_transform(y)

    regr = MultiOutputRegressor(Ridge(random_state=123, fit_intercept=True, positive=False)).fit(X, y_imputed)

    out = regr.predict(X[[0]])
    # params = regr.get_params()
    ridge_coefficients_example = regr.estimators_[0].coef_


hub_gas_prices_pivot = pd.read_pickle("hub_gas_prices_pivot.pkl")
generator_consumption_and_price_df_summary = pd.read_csv(f"generator_consumption_and_price_df_summary.csv")
generator_consumption_and_price_df_summary_pivot = generator_consumption_and_price_df_summary.pivot(index="month", columns="plant_name", values="fuel_cost")


# Impute NaN values in the target
imputer = SimpleImputer(strategy='mean')
generator_consumption_and_price_df_summary_pivot_imputed = imputer.fit_transform(generator_consumption_and_price_df_summary_pivot)

regr = MultiOutputRegressor(Ridge(random_state=123, fit_intercept=True, positive=True, alpha=10.)).fit(hub_gas_prices_pivot, generator_consumption_and_price_df_summary_pivot_imputed)
ridge_coefficients_example = regr.estimators_[0].coef_


# Find the intercepts
intercepts = [estimator.intercept_ for estimator in regr.estimators_]
print("Intercepts:", intercepts)

out = regr.predict(hub_gas_prices_pivot[[0]])

print(18)