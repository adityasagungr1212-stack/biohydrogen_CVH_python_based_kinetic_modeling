import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D

# =========================================================
# 1. KINETIC MODELS
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
# 2. DATA INPUT
# =========================================================

file_path = r'e:\Document 2026\Hydrogen project\Dark\Pyton based kinetic modeling\Step 3.xlsx'

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)

time = df["Time"].values
conditions = df.columns.drop("Time")

# =========================================================
# 3. STORAGE
# =========================================================

results_summary = []
comparison_dfs = {}

# =========================================================
# 4. MODEL FITTING
# =========================================================

for cond in conditions:

    y_true = df[cond].values
    comparison_dfs[cond] = {}
    n_data = len(y_true)

    for model_name, model_func in models.items():

        # ---------------- INITIAL GUESS ----------------
        if model_name == "Gompertz":
            p0 = [np.nanmax(y_true), 1, 1]
            bounds = (0, np.inf)
            num_params = 3

        elif model_name == "First_order":
            p0 = [np.nanmax(y_true), 0.1]
            bounds = (0, np.inf)
            num_params = 2

        elif model_name == "Richards":
            p0 = [np.nanmax(y_true), 1, 1, 1]
            bounds = (0, np.inf)
            num_params = 4

        elif model_name == "Weibull":
            p0 = [np.nanmax(y_true), 0.1, 1]
            bounds = (0, np.inf)
            num_params = 3

        try:
            popt, _ = curve_fit(
                model_func,
                time,
                y_true,
                p0=p0,
                bounds=bounds,
                maxfev=20000
            )

            y_pred = model_func(time, *popt)
            residuals = y_true - y_pred

            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)

            # ---------------- R2 ----------------
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            # ---------------- RMSE & MAE ----------------
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            # ---------------- AIC / BIC ----------------
            aic = n_data * np.log(ss_res / n_data) + 2 * num_params if ss_res > 0 else np.nan
            bic = n_data * np.log(ss_res / n_data) + num_params * np.log(n_data) if ss_res > 0 else np.nan

            # ---------------- STORE RESULT ----------------
            result = {
                "Condition": cond,
                "Model": model_name,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "AIC": aic,
                "BIC": bic
            }

            # ---------------- PARAMETERS ----------------
            if model_name == "Gompertz":
                result.update(dict(zip(["Hmax", "Rmax", "lambda"], popt)))

            elif model_name == "First_order":
                result.update(dict(zip(["Hmax", "k"], popt)))

            elif model_name == "Richards":
                result.update(dict(zip(["Hmax", "a", "b", "nu"], popt)))

            elif model_name == "Weibull":
                result.update(dict(zip(["Hmax", "b", "n"], popt)))

            results_summary.append(result)

            comparison_dfs[cond][model_name] = pd.DataFrame({
                "Time": time,
                "Experimental": y_true,
                "Predicted": y_pred,
                "Residual": residuals
            })

        except Exception as e:
            print(f"FAIL {cond}-{model_name}: {e}")

# =========================================================
# 5. SUMMARY TABLE
# =========================================================

df_summary = pd.DataFrame(results_summary)

# =========================================================
# 6. BEST MODEL PER CONDITION
# =========================================================

df_best = (
    df_summary.sort_values(["Condition", "R2", "RMSE"],
                           ascending=[True, False, True])
    .groupby("Condition")
    .first()
    .reset_index()
)

# =========================================================
# 7. MODEL RANKING
# =========================================================

perf_df = df_summary.groupby("Model")[["R2", "RMSE"]].mean().reset_index()

scaler = MinMaxScaler()
perf_df[["R2_norm", "RMSE_norm"]] = scaler.fit_transform(perf_df[["R2", "RMSE"]])

perf_df["Score"] = perf_df["R2_norm"] + (1 - perf_df["RMSE_norm"])
perf_df = perf_df.sort_values("Score", ascending=False).reset_index(drop=True)

# =========================================================
# 8. SAVE OUTPUT
# =========================================================

output_file = r'e:\Document 2026\Hydrogen project\Dark\Pyton based kinetic modeling\Output_Step3.xlsx'

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_summary.to_excel(writer, index=False, sheet_name="Summary")
    df_best.to_excel(writer, index=False, sheet_name="Best_Model")
    perf_df.to_excel(writer, index=False, sheet_name="Ranking")

print("DONE:", output_file)

# =========================================================
# 9. PLOT FITTING
# =========================================================

colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

for model_name in models.keys():

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for i, cond in enumerate(conditions):

        df_plot = comparison_dfs[cond][model_name]
        c = colors[i]

        time = df_plot["Time"].values
        y_true = df_plot["Experimental"].values

        # Experimental data
        plt.plot(time, y_true, "o", color=c, markersize=5)

        # Model prediction
        plt.plot(df_plot["Time"], df_plot["Predicted"],
                 "-", color=c, linewidth=2.5, label=cond)

    plt.title(f"{model_name} Kinetic Model", fontweight="bold")
    plt.xlabel("Time (h)", fontweight="bold")
    plt.ylabel(r"Cumulative H$_2$ yield (mL H$_2$/gVS)", fontweight="bold")

    ax.grid(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='black',
               linestyle='None', markersize=6, label='Experimental'),
        Line2D([0], [0], color='black', linewidth=2.5, label='Model fit')
    ]

    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        legend_elements.append(
            Line2D([0], [0], color=h.get_color(),
                   linewidth=2.5, label=l)
        )

    plt.legend(handles=legend_elements, loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()

# =========================================================
# WEIBULL SCENARIO CLASSIFICATION
# =========================================================

def weibull_scenario(beta):

    if beta < 1:
        return "Scenario 1: Decelerating rate"

    elif np.isclose(beta, 1, atol=0.05):
        return "Scenario 2: Exponential"

    elif beta <= 3.6:
        return "Scenario 3: Moderate sigmoid"

    else:
        return "Scenario 4: Strong sigmoid"


# hanya Weibull
mask = df_summary["Model"] == "Weibull"

df_summary.loc[mask, "Weibull_Scenario"] = (
    df_summary.loc[mask, "b"]
    .apply(weibull_scenario)
)

# output Weibull
weibull_result = df_summary.loc[
    mask,
    [
        "Condition",
        "Hmax",
        "b",
        "n",
        "R2",
        "RMSE",
        "AIC",
        "Weibull_Scenario"
    ]
]

print("\n===== WEIBULL SCENARIO CLASSIFICATION =====")
print(weibull_result.to_string(index=False))

# =========================================================
# WEIBULL BETA SENSITIVITY ANALYSIS
# =========================================================

condition_target = conditions[0]
# bisa ganti misalnya:
# condition_target = "NiFe"

# ambil parameter optimum Weibull
row = df_summary[
    (df_summary["Condition"] == condition_target) &
    (df_summary["Model"] == "Weibull")
].iloc[0]

Hmax_opt = row["Hmax"]
b_opt = row["b"]
beta_opt = row["n"]

y_true = df[condition_target].values

beta_range = np.linspace(
    max(0.1, beta_opt*0.2),
    beta_opt*3,
    100
)

R2_list = []

for beta in beta_range:

    y_pred = weibull(
        time,
        Hmax_opt,
        b_opt,
        beta
    )

    ss_res = np.sum((y_true-y_pred)**2)
    ss_tot = np.sum((y_true-np.mean(y_true))**2)

    r2 = 1-ss_res/ss_tot

    R2_list.append(r2)

# plot
plt.figure(figsize=(7,5))

plt.plot(
    beta_range,
    R2_list,
    linewidth=3
)

plt.axvline(
    beta_opt,
    linestyle='--',
    linewidth=2
)

plt.xlabel(
    r"Weibull shape parameter ($\beta$)",
    fontweight='bold'
)

plt.ylabel(
    r"$R^2$",
    fontweight='bold'
)

plt.title(
    f"{condition_target}: Sensitivity of β on Weibull fitting",
    fontweight='bold'
)

plt.tight_layout()
plt.show()