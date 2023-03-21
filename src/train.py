import joblib
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_percentage_error,
    explained_variance_score,
    median_absolute_error,
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    PredictionErrorDisplay,
    mean_squared_log_error,
)
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import StackingRegressor
import src.model_dispatcher


def run_training(params):

    # Arrays to store scores
    best_model = []
    nested_scores = np.zeros(params["NUM_TRIALS"])

    # read training and test data
    filename = params["train_file"]
    file_path = params["TRAIN_TEST_FILE_PATH"]
    df_train = pd.read_csv(f"{file_path}\\{filename}")
    filename = params["test_file"]
    df_test = pd.read_csv(f"{file_path}\\{filename}")

    # drop the label column and convert to numpy array
    X_train = df_train.drop(params["target"], axis=1).values
    y_train = df_train[params["target"]].values

    X_test = df_test.drop(params["target"], axis=1).values
    y_test = df_test[params["target"]].values

    # Loop for each trial
    for i in range(params["NUM_TRIALS"]):
        inner_cv = KFold(n_splits=params["n_folds"], shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=params["n_folds"], shuffle=True, random_state=i)

        # Nested CV with parameter optimization
        clf = GridSearchCV(
            estimator=src.model_dispatcher.models[params["model"]],
            param_grid=params["p_grid"],
            cv=inner_cv,
            n_jobs=params["n_jobs"],
            return_train_score=True,
            refit=params["refit_scorer"],
            scoring=params["scoring"],
        )

        # fit the model on training data
        clf.fit(X=X_train, y=y_train)
        nested_score = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv)
        best_model.append(clf)

        nested_scores[i] = nested_score.mean()

    # Plot scores on each trial for nested and non-nested CV
    plt.figure(figsize=(8, 2))
    (nested_line,) = plt.plot(nested_scores, color="b")
    plt.ylabel("score", fontsize="10")
    plt.xlabel("Trial No.", fontsize="10")
    plt.title(
        "Nested Cross Validation",
        x=0.5,
        y=1.1,
        fontsize="11",
    )

    plt.show()
    best_clf = np.where(nested_scores == np.amax(nested_scores))

    # Test the model
    md = params["model"]
    print(f"{md} Testing Performance")
    score_df = create_score_df(
        best_model[best_clf[0][0]], X_train, y_train, X_test, y_test
    )
    model_predict(best_model[best_clf[0][0]], X_test, y_test)

    # save the model
    saved_model = best_model[best_clf[0][0]]
    file_path = params["MODEL_OUTPUT_PATH"]
    joblib.dump(saved_model, f"{file_path}\\best_{md}.bin")
    results = {}
    results["opt_model"] = saved_model
    results["model_metrics"] = score_df
    print(results["model_metrics"])
    return results


def stack_models(params):

    # read training and test data
    filename = params["train_file"]
    file_path = params["TRAIN_TEST_FILE_PATH"]
    df_train = pd.read_csv(f"{file_path}\\{filename}")
    filename = params["test_file"]
    df_test = pd.read_csv(f"{file_path}\\{filename}")

    # drop the label column and convert to numpy array
    X_train = df_train.drop(params["target"], axis=1).values
    y_train = df_train[params["target"]].values

    X_test = df_test.drop(params["target"], axis=1).values
    y_test = df_test[params["target"]].values

    estimators = [
        (est, src.model_dispatcher.models[est]) for est in params["estimators"]
    ]
    final_estimator = (
        params["final_estimator"],
        src.model_dispatcher.models[params["final_estimator"]],
    )
    stacking_regressor = StackingRegressor(
        estimators=estimators, final_estimator=final_estimator[1]
    )
    stack_model = stacking_regressor.fit(X_train, y_train)

    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    axs = np.ravel(axs)

    for ax, (name, est) in zip(
        axs, estimators + [("Stacking Regressor", stacking_regressor)]
    ):
        scorers = {"R2": "r2", "MAE": "neg_mean_absolute_error"}

        start_time = time.time()
        scores = cross_validate(
            est, X_train, y_train, scoring=list(scorers.values()), n_jobs=-1, verbose=0
        )
        elapsed_time = time.time() - start_time

        y_pred = cross_val_predict(est, X_train, y_train, n_jobs=-1, verbose=0)
        scores = {
            key: (
                f"{np.abs(np.mean(scores[f'test_{value}'])):.2f} +- "
                f"{np.std(scores[f'test_{value}']):.2f}"
            )
            for key, value in scorers.items()
        }

        display = PredictionErrorDisplay.from_predictions(
            y_true=y_train,
            y_pred=y_pred,
            kind="actual_vs_predicted",
            ax=ax,
            scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
            line_kwargs={"color": "tab:red"},
        )
        ax.set_title(f"{name}\nEvaluation in {elapsed_time:.2f} seconds")

        for name, score in scores.items():
            ax.plot([], [], " ", label=f"{name}: {score}")
        ax.legend(loc="upper left")

    plt.suptitle("Single predictors versus stacked predictors")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    print("Stack_model Testing Performance")
    score_df = create_score_df(stack_model, X_train, y_train, X_test, y_test)

    # save the model
    file_path = params["MODEL_OUTPUT_PATH"]
    joblib.dump(stack_model, f"{file_path}\\stack_model.bin")
    results = {"stacked_model": stack_model, "model_metrics": score_df}
    print(results["model_metrics"])
    return results


def calc_scores(model, X, y):
    predictions = model.predict(X)
    results = {
        "MSE": mean_squared_error(y, predictions),
        "R2 Score": r2_score(y, predictions),
        "MAPE": mean_absolute_percentage_error(y, predictions),
        "EVS": explained_variance_score(y, predictions),
        "medAE": median_absolute_error(y, predictions),
        "MSLE": mean_squared_log_error(y, predictions),
    }
    return results


def model_predict(model, X, y):

    y_pred = model.predict(X)
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    # Plotting the actual vs predicted values.
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        random_state=0,
    )

    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    plt.show()
    return y_pred


def create_score_df(model, X_train, y_train, X_test, y_test):

    # Calculate Train-Test Performance
    train = calc_scores(model, X_train, y_train)
    test = calc_scores(model, X_test, y_test)

    # Score dictionary
    dicts = [train, test]

    # Merge Training and Test Score
    results = {k: [d[k] for d in dicts] for k in dicts[0]}
    return pd.DataFrame(results, index=["Training", "Testing"])
