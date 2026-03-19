"""
commit 2 — feat: train and tune multi-model sales prediction with Optuna
=========================================================================
File    : scripts/model_training.py
Purpose : Receive engineered arrays from feature_engineering.py,
          train 6 models, tune with Optuna, return the best model
          and predictions — no intermediate files written to disk.

Models:
  1. Ridge Regression      — scaled input, fast linear baseline
  2. Random Forest         — Optuna 30 trials
  3. Gradient Boosting     — Optuna 30 trials
  4. XGBoost               — Optuna 40 trials  (pip install xgboost)
  5. LightGBM              — Optuna 40 trials  (pip install lightgbm)
  6. Stacking Ensemble     — RF + GB + XGB + LGB → Ridge meta-learner

Metrics per model  (all in original $ scale after np.expm1):
  R²  |  CV-R² ± std  |  MAE  |  RMSE  |  MAPE
"""

import warnings
import numpy as np

from sklearn.model_selection  import KFold, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import Ridge
from sklearn.ensemble         import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost  import XGBRegressor;  HAS_XGB = True
except ImportError:
    HAS_XGB = False;  print("[!] pip install xgboost")

try:
    from lightgbm import LGBMRegressor; HAS_LGB = True
except ImportError:
    HAS_LGB = False;  print("[!] pip install lightgbm")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False; print("[!] pip install optuna  (using default params)")

warnings.filterwarnings("ignore")

SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────────
def _mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def _metrics(y_true_dollars, pred_dollars, cv_scores):
    return {
        "R2"       : round(r2_score(y_true_dollars, pred_dollars), 4),
        "CV_R2"    : round(cv_scores.mean(), 4),
        "CV_R2_std": round(cv_scores.std(),  4),
        "MAE"      : round(mean_absolute_error(y_true_dollars, pred_dollars), 2),
        "RMSE"     : round(np.sqrt(mean_squared_error(y_true_dollars, pred_dollars)), 2),
        "MAPE"     : round(_mape(y_true_dollars, pred_dollars), 2),
    }


def run(X_train, X_test, y_train, y_test, y_test_dollars):
    """
    Train all models and return best model + predictions.

    Parameters
    ----------
    X_train, X_test  : np.ndarray  (engineered features)
    y_train, y_test  : np.ndarray  (log_Sales)
    y_test_dollars   : np.ndarray  (original $ Sales for metric reporting)

    Returns
    -------
    best_model   : fitted sklearn estimator
    best_pred_dollars  : np.ndarray  predictions in original $ scale
    all_results  : dict        metrics for every model
    best_name    : str
    """

    print("=" * 55)
    print("  STEP 2 — MODEL TRAINING & TUNING")
    print("=" * 55)

    kf          = KFold(n_splits=5, shuffle=True, random_state=SEED)
    all_results = {}
    all_models  = {}

    # Scale for Ridge only
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    def evaluate(name, model, X_tr, X_te):
        cv = cross_val_score(model, X_tr, y_train,
                             cv=kf, scoring="r2", n_jobs=-1)
        model.fit(X_tr, y_train)
        pred_dollars = np.expm1(model.predict(X_te))
        m = _metrics(y_test_dollars, pred_dollars, cv)
        all_results[name] = m
        all_models[name]  = (model, pred_dollars)
        print(f"  {name:<32}  R²={m['R2']:.4f}  CV-R²={m['CV_R2']:.4f}"
              f"±{m['CV_R2_std']:.4f}  MAE=${m['MAE']:>7,.0f}  MAPE={m['MAPE']:.1f}%")
        return model

    # ── 1. Ridge ─────────────────────────────
    print("\n[1] Ridge Regression")
    evaluate("Ridge", Ridge(alpha=10), X_train_sc, X_test_sc)

    # ── 2. Random Forest ─────────────────────
    print("\n[2] Random Forest")
    if HAS_OPTUNA:
        def rf_obj(trial):
            m = RandomForestRegressor(
                n_estimators     = trial.suggest_int("n_estimators", 100, 500),
                max_depth        = trial.suggest_int("max_depth", 5, 25),
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8),
                max_features     = trial.suggest_float("max_features", 0.3, 1.0),
                random_state=SEED, n_jobs=-1)
            return cross_val_score(m, X_train, y_train,
                                   cv=3, scoring="r2", n_jobs=-1).mean()
        s = optuna.create_study(direction="maximize")
        s.optimize(rf_obj, n_trials=30, show_progress_bar=False)
        rf_p = s.best_params
        print(f"  Best params: {rf_p}")
    else:
        rf_p = dict(n_estimators=300, max_depth=15,
                    min_samples_leaf=3, max_features=0.7)
    rf_model = evaluate("Random Forest",
                         RandomForestRegressor(**rf_p, random_state=SEED, n_jobs=-1),
                         X_train, X_test)

    # ── 3. Gradient Boosting ─────────────────
    print("\n[3] Gradient Boosting")
    if HAS_OPTUNA:
        def gb_obj(trial):
            m = GradientBoostingRegressor(
                n_estimators  = trial.suggest_int("n_estimators", 100, 400),
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth     = trial.suggest_int("max_depth", 3, 8),
                subsample     = trial.suggest_float("subsample", 0.6, 1.0),
                random_state  = SEED)
            return cross_val_score(m, X_train, y_train,
                                   cv=3, scoring="r2", n_jobs=-1).mean()
        s = optuna.create_study(direction="maximize")
        s.optimize(gb_obj, n_trials=30, show_progress_bar=False)
        gb_p = s.best_params
        print(f"  Best params: {gb_p}")
    else:
        gb_p = dict(n_estimators=300, learning_rate=0.05,
                    max_depth=5, subsample=0.8)
    gb_model = evaluate("Gradient Boosting",
                         GradientBoostingRegressor(**gb_p, random_state=SEED),
                         X_train, X_test)

    # ── 4. XGBoost ───────────────────────────
    xgb_p = {}
    if HAS_XGB:
        print("\n[4] XGBoost")
        if HAS_OPTUNA:
            def xgb_obj(trial):
                m = XGBRegressor(
                    n_estimators     = trial.suggest_int("n_estimators", 100, 600),
                    learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    max_depth        = trial.suggest_int("max_depth", 3, 10),
                    subsample        = trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 5.0),
                    reg_lambda       = trial.suggest_float("reg_lambda", 0.0, 5.0),
                    random_state=SEED, verbosity=0, n_jobs=-1)
                return cross_val_score(m, X_train, y_train,
                                       cv=3, scoring="r2", n_jobs=-1).mean()
            s = optuna.create_study(direction="maximize")
            s.optimize(xgb_obj, n_trials=40, show_progress_bar=False)
            xgb_p = s.best_params
            print(f"  Best params: {xgb_p}")
        else:
            xgb_p = dict(n_estimators=300, learning_rate=0.05,
                         max_depth=6, subsample=0.8, colsample_bytree=0.8)
        evaluate("XGBoost",
                 XGBRegressor(**xgb_p, random_state=SEED, verbosity=0, n_jobs=-1),
                 X_train, X_test)

    # ── 5. LightGBM ──────────────────────────
    lgb_p = {}
    if HAS_LGB:
        print("\n[5] LightGBM")
        if HAS_OPTUNA:
            def lgb_obj(trial):
                m = LGBMRegressor(
                    n_estimators  = trial.suggest_int("n_estimators", 100, 600),
                    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                    max_depth     = trial.suggest_int("max_depth", 3, 15),
                    num_leaves    = trial.suggest_int("num_leaves", 20, 200),
                    subsample     = trial.suggest_float("subsample", 0.5, 1.0),
                    reg_alpha     = trial.suggest_float("reg_alpha", 0.0, 5.0),
                    reg_lambda    = trial.suggest_float("reg_lambda", 0.0, 5.0),
                    random_state=SEED, verbose=-1, n_jobs=-1)
                return cross_val_score(m, X_train, y_train,
                                       cv=3, scoring="r2", n_jobs=-1).mean()
            s = optuna.create_study(direction="maximize")
            s.optimize(lgb_obj, n_trials=40, show_progress_bar=False)
            lgb_p = s.best_params
            print(f"  Best params: {lgb_p}")
        else:
            lgb_p = dict(n_estimators=300, learning_rate=0.05,
                         max_depth=7, num_leaves=63, subsample=0.8)
        evaluate("LightGBM",
                 LGBMRegressor(**lgb_p, random_state=SEED, verbose=-1, n_jobs=-1),
                 X_train, X_test)

    # ── 6. Stacking Ensemble ─────────────────
    print("\n[6] Stacking Ensemble  (RF + GB + XGB + LGB → Ridge)")
    base = [
        ("rf", RandomForestRegressor(**rf_p, random_state=SEED, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(**gb_p, random_state=SEED)),
    ]
    if HAS_XGB and xgb_p:
        base.append(("xgb", XGBRegressor(**xgb_p, random_state=SEED,
                                          verbosity=0, n_jobs=-1)))
    if HAS_LGB and lgb_p:
        base.append(("lgb", LGBMRegressor(**lgb_p, random_state=SEED,
                                           verbose=-1, n_jobs=-1)))
    evaluate("Stacking Ensemble",
             StackingRegressor(estimators=base,
                               final_estimator=Ridge(alpha=1.0),
                               cv=5, n_jobs=-1),
             X_train, X_test)

    # ── Pick best by R² ──────────────────────
    best_name  = max(all_results, key=lambda k: all_results[k]["R2"])
    best_model = all_models[best_name][0]
    best_pred  = all_models[best_name][1]

    print(f"\n  🏆  Best model : {best_name}  "
          f"→  R² = {all_results[best_name]['R2']:.4f}")
    print("\n  Model training complete ✓\n")

    return best_model, best_pred, all_results, best_name


if __name__ == "__main__":
    import os
    import feature_engineering as fe
    BASE = os.path.dirname(os.path.abspath(__file__))
    X_train, X_test, y_train, y_test, y_test_dollars, feat_names = fe.run(
        os.path.join(BASE, "dataclean", "orders_cleaned.csv")
    )
    run(X_train, X_test, y_train, y_test, y_test_dollars)