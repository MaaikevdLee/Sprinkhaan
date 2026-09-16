import numpy as np
import mlflow, mlflow.sklearn
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import mlflow.xgboost
from sklearn.svm import SVR


# Default grid — used when the notebook passes grid=None (the "off" template).
DEFAULT_ELASTICNET_GRID = {
    "alpha": np.logspace(-3, 1, 9),
    "l1_ratio": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
}


def fit_elasticnet(X, y, outer_splits, grid=None, n_inner=5, random_state=0):
    """Nested CV, full elastic-net sweep. X already globally scaled (no scaler).

    grid: pass a dict from the notebook to override; leave None to fall back on
    DEFAULT_ELASTICNET_GRID. The grid actually used is printed at run start.
    outer_splits: the shared precomputed (train_idx, val_idx) list.
    """
    grid = grid if grid is not None else DEFAULT_ELASTICNET_GRID
    print(f"[elasticnet] grid in use: {grid}")

    est = ElasticNet(max_iter=10000, random_state=random_state)
    inner = KFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    X = X.reset_index(drop=True); y = y.reset_index(drop=True)

    with mlflow.start_run(run_name="elasticnet"):
        val_r2s, val_rmses, gaps = [], [], []
        for k, (tr, val) in enumerate(outer_splits):
            search = GridSearchCV(est, grid, cv=inner, scoring="r2", refit=True)
            search.fit(X.iloc[tr], y.iloc[tr])
            best = search.best_estimator_

            tr_pred, val_pred = best.predict(X.iloc[tr]), best.predict(X.iloc[val])
            train_r2 = r2_score(y.iloc[tr], tr_pred)
            val_r2 = r2_score(y.iloc[val], val_pred)
            train_rmse = float(np.sqrt(mean_squared_error(y.iloc[tr], tr_pred)))
            val_rmse = float(np.sqrt(mean_squared_error(y.iloc[val], val_pred)))
            gap = train_r2 - val_r2

            with mlflow.start_run(run_name=f"outer_fold_{k}", nested=True):
                mlflow.log_params({f"best_{p}": v for p, v in search.best_params_.items()})
                mlflow.log_metrics({"train_r2": train_r2, "val_r2": val_r2, "gap_r2": gap,
                                    "train_rmse": train_rmse, "val_rmse": val_rmse})
            val_r2s.append(val_r2); val_rmses.append(val_rmse); gaps.append(gap)

        mlflow.log_metrics({
            "val_r2_mean": float(np.mean(val_r2s)), "val_r2_sd": float(np.std(val_r2s)),
            "val_rmse_mean": float(np.mean(val_rmses)), "val_rmse_sd": float(np.std(val_rmses)),
            "gap_r2_mean": float(np.mean(gaps)),
        })

    with mlflow.start_run(run_name="elasticnet_final"):
        final = GridSearchCV(est, grid, cv=inner, scoring="r2", refit=True)
        final.fit(X, y)
        mlflow.log_params({f"best_{p}": v for p, v in final.best_params_.items()})
        mlflow.sklearn.log_model(final.best_estimator_, name="elasticnet")

    return {"val_r2_mean": float(np.mean(val_r2s)), "val_r2_sd": float(np.std(val_r2s)),
            "val_rmse_mean": float(np.mean(val_rmses)), "gap_r2_mean": float(np.mean(gaps))}


######## Linear ####


def fit_linear(X, y, outer_splits, random_state=0):
    """Nested CV, OLS baseline. X already globally scaled (no scaler).

    No hyperparameters -> no inner CV, no GridSearchCV: each outer fold just fits
    LinearRegression on its train part. The outer loop still gives an honest
    unseen-data estimate, so this is the reference the tuned models must beat.

    outer_splits: the shared precomputed (train_idx, val_idx) list — same folds,
    same patients, as every other fit_* using it.
    random_state kept only for signature parity; OLS is deterministic and ignores it.
    """
    X = X.reset_index(drop=True); y = y.reset_index(drop=True)

    with mlflow.start_run(run_name="linear_ols"):
        val_r2s, val_rmses, gaps = [], [], []
        for k, (tr, val) in enumerate(outer_splits):
            model = LinearRegression().fit(X.iloc[tr], y.iloc[tr])

            tr_pred, val_pred = model.predict(X.iloc[tr]), model.predict(X.iloc[val])
            train_r2 = r2_score(y.iloc[tr], tr_pred)
            val_r2 = r2_score(y.iloc[val], val_pred)
            train_rmse = float(np.sqrt(mean_squared_error(y.iloc[tr], tr_pred)))
            val_rmse = float(np.sqrt(mean_squared_error(y.iloc[val], val_pred)))
            gap = train_r2 - val_r2

            with mlflow.start_run(run_name=f"outer_fold_{k}", nested=True):
                mlflow.log_metrics({"train_r2": train_r2, "val_r2": val_r2, "gap_r2": gap,
                                    "train_rmse": train_rmse, "val_rmse": val_rmse})
            val_r2s.append(val_r2); val_rmses.append(val_rmse); gaps.append(gap)

        mlflow.log_metrics({
            "val_r2_mean": float(np.mean(val_r2s)), "val_r2_sd": float(np.std(val_r2s)),
            "val_rmse_mean": float(np.mean(val_rmses)), "val_rmse_sd": float(np.std(val_rmses)),
            "gap_r2_mean": float(np.mean(gaps)),
        })

    with mlflow.start_run(run_name="linear_ols_final"):
        final = LinearRegression().fit(X, y)
        mlflow.sklearn.log_model(final, name="linear_ols")

    return {"val_r2_mean": float(np.mean(val_r2s)), "val_r2_sd": float(np.std(val_r2s)),
            "val_rmse_mean": float(np.mean(val_rmses)), "gap_r2_mean": float(np.mean(gaps))}


##### RANDOM FOREST ####

# Default grid — used when the notebook passes grid=None ("off"). Kept small
# because per-drug cohorts are tiny; widen in the notebook if val stays low.
DEFAULT_RF_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [None, 4, 8],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", 1.0],
}


def fit_randomforest(X, y, outer_splits, grid=None, n_inner=5, random_state=0):
    """Nested CV, random-forest regressor. X scaling irrelevant (trees).

    grid: dict from the notebook to override; None falls back on DEFAULT_RF_GRID.
    The grid actually used is printed at run start.
    outer_splits: the shared precomputed (train_idx, val_idx) list.
    """
    grid = grid if grid is not None else DEFAULT_RF_GRID
    print(f"[randomforest] grid in use: {grid}")

    est = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    inner = KFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    X = X.reset_index(drop=True); y = y.reset_index(drop=True)

    with mlflow.start_run(run_name="randomforest"):
        val_r2s, val_rmses, gaps = [], [], []
        for k, (tr, val) in enumerate(outer_splits):
            search = GridSearchCV(est, grid, cv=inner, scoring="r2", refit=True, n_jobs=-1)
            search.fit(X.iloc[tr], y.iloc[tr])
            best = search.best_estimator_

            tr_pred, val_pred = best.predict(X.iloc[tr]), best.predict(X.iloc[val])
            train_r2 = r2_score(y.iloc[tr], tr_pred)
            val_r2 = r2_score(y.iloc[val], val_pred)
            train_rmse = float(np.sqrt(mean_squared_error(y.iloc[tr], tr_pred)))
            val_rmse = float(np.sqrt(mean_squared_error(y.iloc[val], val_pred)))
            gap = train_r2 - val_r2

            with mlflow.start_run(run_name=f"outer_fold_{k}", nested=True):
                mlflow.log_params({f"best_{p}": v for p, v in search.best_params_.items()})
                mlflow.log_metrics({"train_r2": train_r2, "val_r2": val_r2, "gap_r2": gap,
                                    "train_rmse": train_rmse, "val_rmse": val_rmse})
            val_r2s.append(val_r2); val_rmses.append(val_rmse); gaps.append(gap)

        mlflow.log_metrics({
            "val_r2_mean": float(np.mean(val_r2s)), "val_r2_sd": float(np.std(val_r2s)),
            "val_rmse_mean": float(np.mean(val_rmses)), "val_rmse_sd": float(np.std(val_rmses)),
            "gap_r2_mean": float(np.mean(gaps)),
        })

    with mlflow.start_run(run_name="randomforest_final"):
        final = GridSearchCV(est, grid, cv=inner, scoring="r2", refit=True, n_jobs=-1)
        final.fit(X, y)
        mlflow.log_params({f"best_{p}": v for p, v in final.best_params_.items()})
        mlflow.sklearn.log_model(final.best_estimator_, name="randomforest")

    return {"val_r2_mean": float(np.mean(val_r2s)), "val_r2_sd": float(np.std(val_r2s)),
            "val_rmse_mean": float(np.mean(val_rmses)), "gap_r2_mean": float(np.mean(gaps))}



# Default grid — used when the notebook passes grid=None ("off"). Deliberately
# small and regularised: shallow trees, slow learning, subsampling. On tiny
# cohorts a deep/greedy XGBoost memorises the train set.
DEFAULT_XGB_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [2, 3],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1.0, 5.0],
}


def fit_xgboost(X, y, outer_splits, grid=None, n_inner=5, random_state=0):
    """Nested CV, XGBoost regressor. X scaling irrelevant (trees).

    grid: dict from the notebook to override; None falls back on DEFAULT_XGB_GRID.
    The grid actually used is printed at run start.
    outer_splits: the shared precomputed (train_idx, val_idx) list.
    No early stopping: it needs a held-out eval set, which would eat into the
    fold and break the clean nested structure. Tune n_estimators via the grid.
    """
    grid = grid if grid is not None else DEFAULT_XGB_GRID
    print(f"[xgboost] grid in use: {grid}")

    est = XGBRegressor(random_state=random_state, n_jobs=-1,
                       objective="reg:squarederror", tree_method="hist")
    inner = KFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    X = X.reset_index(drop=True); y = y.reset_index(drop=True)

    with mlflow.start_run(run_name="xgboost"):
        val_r2s, val_rmses, gaps = [], [], []
        for k, (tr, val) in enumerate(outer_splits):
            search = GridSearchCV(est, grid, cv=inner, scoring="r2", refit=True, n_jobs=-1)
            search.fit(X.iloc[tr], y.iloc[tr])
            best = search.best_estimator_

            tr_pred, val_pred = best.predict(X.iloc[tr]), best.predict(X.iloc[val])
            train_r2 = r2_score(y.iloc[tr], tr_pred)
            val_r2 = r2_score(y.iloc[val], val_pred)
            train_rmse = float(np.sqrt(mean_squared_error(y.iloc[tr], tr_pred)))
            val_rmse = float(np.sqrt(mean_squared_error(y.iloc[val], val_pred)))
            gap = train_r2 - val_r2

            with mlflow.start_run(run_name=f"outer_fold_{k}", nested=True):
                mlflow.log_params({f"best_{p}": v for p, v in search.best_params_.items()})
                mlflow.log_metrics({"train_r2": train_r2, "val_r2": val_r2, "gap_r2": gap,
                                    "train_rmse": train_rmse, "val_rmse": val_rmse})
            val_r2s.append(val_r2); val_rmses.append(val_rmse); gaps.append(gap)

        mlflow.log_metrics({
            "val_r2_mean": float(np.mean(val_r2s)), "val_r2_sd": float(np.std(val_r2s)),
            "val_rmse_mean": float(np.mean(val_rmses)), "val_rmse_sd": float(np.std(val_rmses)),
            "gap_r2_mean": float(np.mean(gaps)),
        })

    with mlflow.start_run(run_name="xgboost_final"):
        final = GridSearchCV(est, grid, cv=inner, scoring="r2", refit=True, n_jobs=-1)
        final.fit(X, y)
        mlflow.log_params({f"best_{p}": v for p, v in final.best_params_.items()})
        mlflow.xgboost.log_model(final.best_estimator_, name="xgboost")

    return {"val_r2_mean": float(np.mean(val_r2s)), "val_r2_sd": float(np.std(val_r2s)),
            "val_rmse_mean": float(np.mean(val_rmses)), "gap_r2_mean": float(np.mean(gaps))}



# Default grid — used when the notebook passes grid=None ("off"). C = tolerance
# for error, gamma = reach of each point's influence, epsilon = the no-penalty
# tube width. RBF kernel only here; add "kernel" to the grid to compare.
DEFAULT_SVR_GRID = {
    "C": [0.1, 1.0, 10.0],
    "gamma": ["scale", 0.01, 0.1],
    "epsilon": [0.01, 0.1],
}


def fit_svr(X, y, outer_splits, grid=None, n_inner=5, random_state=0):
    """Nested CV, support vector regression (RBF). X MUST be scaled — it is,
    globally, upstream, so no scaler here (same pre-scaled regime as the rest).

    grid: dict from the notebook to override; None falls back on DEFAULT_SVR_GRID.
    The grid actually used is printed at run start.
    outer_splits: the shared precomputed (train_idx, val_idx) list.
    random_state kept for signature parity; SVR is deterministic.
    """
    grid = grid if grid is not None else DEFAULT_SVR_GRID
    print(f"[svr] grid in use: {grid}")

    est = SVR(kernel="rbf")
    inner = KFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    X = X.reset_index(drop=True); y = y.reset_index(drop=True)

    with mlflow.start_run(run_name="svr"):
        val_r2s, val_rmses, gaps = [], [], []
        for k, (tr, val) in enumerate(outer_splits):
            search = GridSearchCV(est, grid, cv=inner, scoring="r2", refit=True, n_jobs=-1)
            search.fit(X.iloc[tr], y.iloc[tr])
            best = search.best_estimator_

            tr_pred, val_pred = best.predict(X.iloc[tr]), best.predict(X.iloc[val])
            train_r2 = r2_score(y.iloc[tr], tr_pred)
            val_r2 = r2_score(y.iloc[val], val_pred)
            train_rmse = float(np.sqrt(mean_squared_error(y.iloc[tr], tr_pred)))
            val_rmse = float(np.sqrt(mean_squared_error(y.iloc[val], val_pred)))
            gap = train_r2 - val_r2

            with mlflow.start_run(run_name=f"outer_fold_{k}", nested=True):
                mlflow.log_params({f"best_{p}": v for p, v in search.best_params_.items()})
                mlflow.log_metrics({"train_r2": train_r2, "val_r2": val_r2, "gap_r2": gap,
                                    "train_rmse": train_rmse, "val_rmse": val_rmse})
            val_r2s.append(val_r2); val_rmses.append(val_rmse); gaps.append(gap)

        mlflow.log_metrics({
            "val_r2_mean": float(np.mean(val_r2s)), "val_r2_sd": float(np.std(val_r2s)),
            "val_rmse_mean": float(np.mean(val_rmses)), "val_rmse_sd": float(np.std(val_rmses)),
            "gap_r2_mean": float(np.mean(gaps)),
        })

    with mlflow.start_run(run_name="svr_final"):
        final = GridSearchCV(est, grid, cv=inner, scoring="r2", refit=True, n_jobs=-1)
        final.fit(X, y)
        mlflow.log_params({f"best_{p}": v for p, v in final.best_params_.items()})
        mlflow.sklearn.log_model(final.best_estimator_, name="svr")

    return {"val_r2_mean": float(np.mean(val_r2s)), "val_r2_sd": float(np.std(val_r2s)),
            "val_rmse_mean": float(np.mean(val_rmses)), "gap_r2_mean": float(np.mean(gaps))}