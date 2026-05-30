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

file_path = r'e:\Document 2026\Hydrogen project\Dark\Biohydrogen from CVH\data_\Step3_Chemical_additives.xlsx'

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

output_file = r'e:\Document 2026\Hydrogen project\Dark\Biohydrogen from CVH\result_\Step3_Chemical_additives.xlsx'

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_summary.to_excel(writer, index=False, sheet_name="Summary")
    df_best.to_excel(writer, index=False, sheet_name="Best_Model")
    perf_df.to_excel(writer, index=False, sheet_name="Ranking")

print("DONE:", output_file)
