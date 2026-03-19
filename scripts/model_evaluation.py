import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── plot style ────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
ORANGE = "#F59E0B"
GREEN  = "#10B981"
RED    = "#EF4444"
GRAY   = "#6B7280"
BG     = "#F9FAFB"

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : BG,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 12,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 10,
})


def _mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def run(best_model, best_pred_dollars, y_test_dollars,
        all_results, best_name, feature_names, output_dir):
    """
    Evaluate, plot and save ONLY sales_predictions_final.csv.

    Parameters
    ----------
    best_model      : fitted estimator
    best_pred_dollars     : np.ndarray  predictions in original $ scale
    y_test_dollars  : np.ndarray  actual Sales in original $ scale
    all_results     : dict        metrics from model_training.py
    best_name       : str
    feature_names   : list[str]
    output_dir      : str         path to scripts/dataclean/
    """

    print("=" * 55)
    print("  STEP 3 — EVALUATION & VISUALIZATION")
    print("=" * 55)

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ── Final metrics ─────────────────────────
    residuals = y_test_dollars - best_pred_dollars
    r2   = r2_score(y_test_dollars, best_pred_dollars)
    mae  = mean_absolute_error(y_test_dollars, best_pred_dollars)
    rmse = np.sqrt(mean_squared_error(y_test_dollars, best_pred_dollars))
    mp   = _mape(y_test_dollars, best_pred_dollars)

    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │  FINAL RESULTS  —  {best_name:<21}│")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │  R²     :  {r2:.4f}  ({r2*100:.1f}% variance explained)  │")
    print(f"  │  MAE    : ${mae:>9,.2f}  avg error per order     │")
    print(f"  │  RMSE   : ${rmse:>9,.2f}                          │")
    print(f"  │  MAPE   :  {mp:>6.2f}%  avg % error per order   │")
    print(f"  └──────────────────────────────────────────┘")

    # ── SAVE — only final predictions CSV ────
    out_df = pd.DataFrame({
        "Actual_Sales"   : y_test_dollars,
        "Predicted_Sales": best_pred_dollars,
        "Residual_$"     : residuals,
        "APE_%"          : np.abs(residuals) / np.where(
                               y_test_dollars != 0, y_test_dollars, np.nan) * 100,
    })
    out_path = os.path.join(output_dir, "sales_predictions_final.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n  ✓ sales_predictions_final.csv saved  →  {out_path}")
    print(f"    (only this CSV is tracked by Git — all others are in .gitignore)")

    # ── Plot 1 — Actual vs Predicted ──────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test_dollars, best_pred_dollars, alpha=0.35, s=18,
               color=BLUE, edgecolors="none", label="Orders")
    lim = max(y_test_dollars.max(), best_pred_dollars.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=RED, lw=1.8,
            linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual Sales ($)")
    ax.set_ylabel("Predicted Sales ($)")
    ax.set_title(f"Actual vs Predicted Sales\n{best_name}  |  R² = {r2:.4f}")
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "01_actual_vs_predicted.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2 — Residuals ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(best_pred_dollars, residuals, alpha=0.35, s=16,
                    color=ORANGE, edgecolors="none")
    axes[0].axhline(0, color=RED, lw=1.8, linestyle="--")
    axes[0].set_xlabel("Predicted Sales ($)")
    axes[0].set_ylabel("Residual ($)")
    axes[0].set_title("Residual vs Predicted\n(random scatter = unbiased model)")
    axes[0].xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))

    axes[1].hist(residuals, bins=50, color=BLUE, edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color=RED, lw=1.8, linestyle="--", label="Zero error")
    axes[1].axvline(residuals.mean(), color=GREEN, lw=1.5,
                    linestyle=":", label=f"Mean = ${residuals.mean():,.0f}")
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution\n(centred on 0 = no systematic bias)")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "02_residuals.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3 — Model comparison ────────────
    names   = list(all_results.keys())
    r2s     = [all_results[n]["R2"]   for n in names]
    mapes   = [all_results[n]["MAPE"] for n in names]
    maes    = [all_results[n]["MAE"]  for n in names]
    colors  = [GREEN if n == best_name else BLUE for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, vals, label, fmt in zip(
        axes,
        [r2s,  mapes,         maes],
        ["R²", "MAPE (%)",    "MAE ($)"],
        ["{:.3f}", "{:.1f}%", "${:,.0f}"],
    ):
        bars = ax.barh(names, vals, color=colors, height=0.55)
        ax.set_xlabel(label)
        ax.set_title(f"{label}\n(green = best model)")
        for bar, v in zip(bars, vals):
            ax.text(v * 1.01, bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "03_model_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 4 — Feature importance ──────────
    fig, ax = plt.subplots(figsize=(8, 6))
    imp_model = None
    if hasattr(best_model, "feature_importances_"):
        imp_model = best_model
    elif hasattr(best_model, "estimators_"):        # StackingRegressor
      for item in best_model.estimators_:
        est = item[-1]  # last element is always the fitted estimator
        if hasattr(est, "feature_importances_"):
            imp_model = est; break 

    if imp_model is not None:
        fi = (pd.Series(imp_model.feature_importances_, index=feature_names)
              .sort_values().tail(12))
        bar_colors = [GREEN if i == len(fi) - 1 else BLUE
                      for i in range(len(fi))]
        fi.plot(kind="barh", ax=ax, color=bar_colors)
        ax.set_xlabel("Importance Score")
        ax.set_title("Top 12 Features Driving Sales")
        for i, v in enumerate(fi.values):
            ax.text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Feature importance not available\nfor this model type",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color=GRAY)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "04_feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ 4 diagnostic plots saved  →  {plot_dir}/")
    print(f"\n{'='*55}")
    print(f"  PIPELINE COMPLETE ✓")
    print(f"  Best model : {best_name}")
    print(f"  R²         : {r2:.4f}")
    print(f"  MAE        : ${mae:,.2f}")
    print(f"  MAPE       : {mp:.1f}%")
    print(f"{'='*55}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import feature_engineering as fe
    import model_training as mt

    BASE      = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR  = os.path.join(BASE, "dataclean")
    INPUT     = os.path.join(DATA_DIR, "orders_cleaned.csv")

    # Step 1
    X_train, X_test, y_train, y_test, y_test_dollars, feat_names = fe.run(INPUT)

    # Step 2
    best_model, best_pred, all_results, best_name = mt.run(
        X_train, X_test, y_train, y_test, y_test_dollars)

    # Step 3
    run(best_model, best_pred, y_test_dollars,
        all_results, best_name, feat_names, DATA_DIR)