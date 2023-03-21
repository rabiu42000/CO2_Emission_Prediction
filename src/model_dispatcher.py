from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV


models = {
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge(),
    "sgd_regression": SGDRegressor(),
    "adaboost_regression": AdaBoostRegressor(),
    "gboost_regression": GradientBoostingRegressor(),
    "rf_regression": RandomForestRegressor(),
    "lgb_regression": LGBMRegressor(),
    "ridge_cv": RidgeCV(),
}
