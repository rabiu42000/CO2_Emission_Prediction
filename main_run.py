import time
import os
import src.train

# define parameter dict
params = {}
params["models"] = [
    "ridge_regression",
    "gboost_regression",
    "rf_regression",
    "lgb_regression",
]
params["stack_regressors"] = True
params["estimators"] = ["ridge_regression", "rf_regression", "gboost_regression"]
params["final_estimator"] = "ridge_cv"
params["target"] = "CO2_Emissions"
params["n_jobs"] = 5
params["n_folds"] = 4
# Number of random trials
params["NUM_TRIALS"] = 100

params["train_file"] = "CO2_emission_train.csv"
params["test_file"] = "CO2_emission_test.csv"

params["scoring"] = {
    "R2_Score": "r2",
    "MSE": "neg_mean_squared_error",
    "MSLE": "neg_mean_squared_log_error",
    "MAPE": "neg_mean_absolute_percentage_error",
    "EVS": "explained_variance",
    "MedAE": "neg_median_absolute_error",
}

params["refit_scorer"] = "R2_Score"
params["p_grids"] = {
    "ridge_regression": {
        "alpha": [0.1, 1, 10, 100],
        "fit_intercept": [True, False],
    },
    "adaboost_regression": {
        "learning_rate": [0.01, 0.1, 1],
        "n_estimators": [10, 100],
    },
    "gboost_regression": {
        "learning_rate": [0.01, 0.1, 1],
        "n_estimators": [10, 100],
    },
    "rf_regression": {"n_estimators": [20, 40, 80, 160], "max_depth": [1, 10]},
    "lgb_regression": {"learning_rate": [0.01, 0.1, 1], "n_estimators": [10, 100]},
}

# set output files path
list_of_files = list(os.walk(os.getcwd()))
root = list_of_files[0][0]
params["TRAIN_TEST_FILE_PATH"] = os.path.join(root, "input")
params["MODEL_OUTPUT_PATH"] = os.path.join(root, "models")


os.system("cls")
if __name__ == "__main__":

    # create k-folds for training
    if params["n_folds"] < 2:
        params["n_folds"] = 2

    for mdl in params["models"]:
        params["model"] = mdl
        params["p_grid"] = params["p_grids"][mdl]
        # train and save validated model
        print(f"Training {mdl} model")
        start_time = time.time()
        src.train.run_training(params)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Time to run {mdl} gridsearch is: {run_time/60} mins")

    if params["stack_regressors"] == True:
        print("Training Stacked models")
        start_time = time.time()
        src.train.stack_models(params)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Time to run Stacked models is: {run_time/60} mins")
