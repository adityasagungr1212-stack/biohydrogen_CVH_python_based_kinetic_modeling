import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# =========================================================
# 1. MODELS
# =========================================================

def gompertz(t, Hmax, Rmax, lam):
    e = np.exp(1)
    return Hmax * np.exp(-np.exp((Rmax * e / Hmax) * (lam - t) + 1))

def first_order(t, Hmax, k):
    return Hmax * (1 - np.exp(-k * t))

def richards(t, Hmax, a, b, nu):
    return Hmax * (1 + a * np.exp(-b * t))**(-1 / nu)

def weibull(t, Hmax, b, n):
    return Hmax * (1 - np.exp(-(b * t)**n))

models = {
    "Gompertz": gompertz,
    "First_order": first_order,
    "Richards": richards,
    "Weibull": weibull
}

# =========================================================
# 2. PARAMETER MAP
# =========================================================

param_names = {
    "Gompertz": ["Hmax", "Rmax", "lambda"],
    "First_order": ["Hmax", "k"],
    "Richards": ["Hmax", "a", "b", "nu"],
    "Weibull": ["Hmax", "b", "n"]
}

all_param_cols = ["Hmax", "Rmax", "lambda", "k", "a", "b", "n", "nu"]

# =========================================================
# 3. DATA
# =========================================================

file_path = r'e:\Document 2026\Hydrogen project\Dark\Biohydrogen from CVH\data_\Step1_Concentration_of_CVH.xlsx'
df = pd.read_excel(file_path)

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)

time = df['Time'].values
treatments = df.columns.drop('Time')

# =========================================================
# 4. STORAGE
# =========================================================

results_summary = []
comparison_dfs = {}

# =========================================================
# 5. FITTING LOOP
# =========================================================

for treatment in treatments:

    y_true = df[treatment].values
    comparison_dfs[treatment] = {}
    n_data = len(y_true)

    for model_name, model_func in models.items():

        # =====================================================
        # SKIP IF NO HYDROGEN PRODUCTION
        # =====================================================
        if np.nanmax(y_true) <= 0:

            result = {
                "Treatment": treatment,
                "Model": model_name,
                "R2": np.nan,
                "R2adj": np.nan,
                "RMSE": np.nan,
                "AIC": np.nan,
            }

            # fill parameter columns with NaN
            for col in all_param_cols:
                result[col] = np.nan

            results_summary.append(result)

            print(f"SKIPPED {treatment}-{model_name} (no H2 production)")

            continue

        # =====================================================
        # INITIAL PARAMETERS
        # =====================================================
        if model_name == 'Gompertz':
            p0 = [np.nanmax(y_true), 1, 1]
            bounds = (0, [np.inf]*3)

        elif model_name == 'First_order':
            p0 = [np.nanmax(y_true), 0.1]
            bounds = (0, [np.inf]*2)

        elif model_name == 'Richards':
            p0 = [np.nanmax(y_true), 1, 0.5, 1]
            bounds = (0, [np.inf]*4)

        elif model_name == 'Weibull':
            p0 = [np.nanmax(y_true), 0.1, 1]
            bounds = (0, [np.inf]*3)

        # =====================================================
        # MODEL FITTING
        # =====================================================
        try:

            popt, _ = curve_fit(
                model_func,
                time,
                y_true,
                p0=p0,
                bounds=bounds,
                maxfev=20000
            )

            # =================================================
            # PREDICTION
            # =================================================
            y_pred = model_func(time, *popt)

            residuals = y_true - y_pred

            # =================================================
            # STATISTICS
            # =================================================
            ss_res = np.sum(residuals**2)

            ss_tot = np.sum(
                (y_true - np.mean(y_true))**2
            )

            # R²
            if ss_tot == 0:
                r2 = np.nan
            else:
                r2 = 1 - ss_res / ss_tot

            # Adjusted R²
            if n_data <= len(popt) + 1:
                r2_adj = np.nan
            else:
                r2_adj = (
                    1 - (1 - r2) *
                    (n_data - 1) /
                    (n_data - len(popt) - 1)
                )

            # RMSE
            rmse = np.sqrt(
                mean_squared_error(y_true, y_pred)
            )

            # AIC
            if ss_res <= 0:
                aic = np.nan
            else:
                aic = (
                    n_data * np.log(ss_res / n_data)
                    + 2 * len(popt)
                )


            # =================================================
            # SAVE RESULT
            # =================================================
            result = {
                "Treatment": treatment,
                "Model": model_name,
                "R2": r2,
                "R2adj": r2_adj,
                "RMSE": rmse,
                "AIC": aic,
            }

            # save fitted parameters
            for name, value in zip(
                param_names[model_name],
                popt
            ):
                result[name] = value

            # fill unused parameters
            for col in all_param_cols:
                if col not in result:
                    result[col] = np.nan

            results_summary.append(result)

            # =================================================
            # SAVE COMPARISON DATAFRAME
            # =================================================
            comparison_dfs[treatment][model_name] = pd.DataFrame({
                "Time": time,
                "Experimental": y_true,
                "Predicted": y_pred,
                "Residual": residuals
            })

            print(f"OK {treatment}-{model_name}")

        except Exception as e:

            print(f"FAIL {treatment}-{model_name}: {e}")


# =========================================================
# 6. SUMMARY TABLE
# =========================================================

df_summary = pd.DataFrame(results_summary)


# =========================================================
# 7. BEST MODEL
# =========================================================

df_best = (
    df_summary
    .sort_values(
        ["Treatment", "R2adj", "RMSE"],
        ascending=[True, False, True]
    )
    .groupby("Treatment")
    .first()
    .reset_index()
)


# =========================================================
# 8. HEATMAP VISUALIZATION
# =========================================================

# limit adjusted R² maximum
df_summary["Adj_R2_plot"] = (
    df_summary["R2adj"]
    .clip(upper=0.999)
)


def plot_heatmap(metric, title, cmap):

    plt.figure(figsize=(8, 6))

    pivot = df_summary.pivot(
        index="Model",
        columns="Treatment",
        values=metric
    )

    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=0.3,
        linecolor="black",
        annot_kws={"fontsize": 12},
        cbar_kws={"shrink": 0.6},
        square=True
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Concentration of CVH",
        fontweight="bold",
        fontsize=14
    )

    ax.set_ylabel(
        "Model",
        fontweight="bold",
        fontsize=14
    )

    ax.tick_params(
        axis='x',
        rotation=45,
        labelsize=11
    )

    ax.tick_params(
        axis='y',
        rotation=0,
        labelsize=11
    )

    plt.tight_layout()
    plt.show()


# =========================================================
# PLOT 1 - Adjusted R²
# =========================================================

plot_heatmap(
    "Adj_R2_plot",
    "Adjusted R² Comparison",
    "viridis"
)

# =========================================================
# PLOT 2 - RMSE
# =========================================================

plot_heatmap(
    "RMSE",
    "RMSE Comparison",
    "Reds"
)

# =========================================================
# PLOT 3 - AIC
# =========================================================

plot_heatmap(
    "AIC",
    "AIC Comparison",
    "coolwarm"
)


# =========================================================
# 9. RANKING MODEL SCORE (FIXED)
# =========================================================

from sklearn.preprocessing import MinMaxScaler

# 1. Aggregate performance per model
perf_df = df_summary.groupby("Model")[["R2", "RMSE"]].mean().reset_index()

# 2. Normalize metrics (important because scale berbeda)
scaler = MinMaxScaler()
perf_df[["R2_norm", "RMSE_norm"]] = scaler.fit_transform(perf_df[["R2", "RMSE"]])

# 3. Flip RMSE (karena makin kecil makin bagus)
perf_df["RMSE_score"] = 1 - perf_df["RMSE_norm"]

# 4. Final composite score
perf_df["Score"] = perf_df["R2_norm"] + perf_df["RMSE_score"]

# 5. Sort (higher = better)
perf_df = perf_df.sort_values("Score", ascending=False).reset_index(drop=True)

# 6. Category (FIXED logic)
perf_df["Category"] = [
    "Best" if i == 0 else "Good" if i <= 2 else "Poor"
    for i in range(len(perf_df))
]

perf_df

# =========================================================
# 10. SAVE OUTPUT
# =========================================================

output_file = r'e:\Document 2026\Hydrogen project\Dark\Biohydrogen from CVH\result_\Step1_Concentration_of_CVH.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_summary.to_excel(writer, index=False, sheet_name="Summary")
    df_best.to_excel(writer, index=False, sheet_name="Best_Model")
    perf_df.to_excel(writer, index=False, sheet_name="Ranking")

print("DONE:", output_file)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

colors = plt.cm.tab10(np.linspace(0, 1, len(treatments)))

for model_name in models.keys():

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for i, treatment in enumerate(treatments):

        c = colors[i]

        # =========================
        # CEK DATA MODEL
        # =========================
        if model_name in comparison_dfs[treatment]:
            df_plot = comparison_dfs[treatment][model_name]

            time = df_plot["Time"].values
            y_true = df_plot["Experimental"].values
            y_pred = df_plot["Predicted"].values

        else:
            # =========================
            # FALLBACK = 0 LINE
            # =========================
            print(f"Fallback 0-line: {treatment} - {model_name}")

            time = time  # pakai global time
            y_true = df[treatment].values
            y_pred = np.zeros_like(time)

        # =========================
        # Experimental data
        # =========================
        plt.plot(
            time,
            y_true,
            "o",
            color=c,
            markersize=5
        )

        # =========================
        # Model line (REAL or ZERO)
        # =========================
        plt.plot(
            time,
            y_pred,
            "-",
            color=c,
            linewidth=2.5,
            label=treatment
        )
        

    # =========================
    # LABELS
    # =========================
    plt.title(f"Kinetic Model: {model_name}", fontweight="bold", fontsize=14)
    plt.xlabel("Time (h)", fontweight="bold", fontsize=13)
    plt.ylabel(r"Cumulative hydrogen yield (mL H$_2$/g VS)", fontweight="bold", fontsize=13)

    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', width=1.5, labelsize=12)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.grid(False)

    plt.legend(loc="best", frameon=False, fontsize=11)
    plt.tight_layout()
    plt.show()