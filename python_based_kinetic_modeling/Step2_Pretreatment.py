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

file_path = r'e:\Document 2026\Hydrogen project\Dark\Biohydrogen from CVH\data_\Step2_Pretreatment.xlsx'
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

        # ---------------- INIT ----------------
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

            r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

            r2_adj = np.nan if n_data <= len(popt) + 1 else \
                1 - (1 - r2) * (n_data - 1) / (n_data - len(popt) - 1)

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            aic = np.nan if ss_res <= 0 else n_data * np.log(ss_res / n_data) + 2 * len(popt)
            bic = np.nan if ss_res <= 0 else n_data * np.log(ss_res / n_data) + len(popt) * np.log(n_data)

            result = {
                "Treatment": treatment,
                "Model": model_name,
                "R2": r2,
                "R2adj": r2_adj,
                "RMSE": rmse,
                "AIC": aic,
                "BIC": bic
            }

            for name, value in zip(param_names[model_name], popt):
                result[name] = value

            for col in all_param_cols:
                if col not in result:
                    result[col] = np.nan

            results_summary.append(result)

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
    df_summary.sort_values(["Treatment", "R2adj", "RMSE"],
                           ascending=[True, False, True])
    .groupby("Treatment")
    .first()
    .reset_index()
)

# =========================================================
# 8. HEATMAP VISUALIZATION - COLORBAR TINGGI SAMA DENGAN HEATMAP
# =========================================================

import matplotlib.pyplot as plt
import seaborn as sns

df_summary["Adj_R2_plot"] = df_summary["R2adj"].clip(upper=0.999)

# Common settings dengan penyesuaian colorbar yang lebih presisi
common_heatmap_kws = dict(
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.3,
    linecolor="black",
    annot_kws={"fontsize": 9.5},
    cbar_kws={
        "shrink": 0.4,      # Full height
        "aspect": 15,       # Semakin kecil = lebih ramping & tinggi
        "pad": 0.04,
        "location": "right"
    }
)

def create_heatmap(data, title, cmap):
    plt.figure(figsize=(13, 8))   # Ukuran figure yang lebih baik
    
    ax = sns.heatmap(
        data,
        cmap=cmap,
        **common_heatmap_kws
    )
    
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Effect of pretreatment", fontsize=13, fontweight="bold")
    plt.ylabel("Model", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Cara paling efektif agar colorbar pas tingginya
    plt.tight_layout(rect=[0, 0, 0.92, 1])   # Memberi ruang untuk colorbar
    
    plt.show()

# =========================================================
# 1. ADJUSTED R²
# =========================================================
pivot_r2 = df_summary.pivot_table(index="Model", columns="Treatment", 
                                  values="Adj_R2_plot", aggfunc='mean')
create_heatmap(pivot_r2, "Adjusted R² Comparison", "viridis")

# =========================================================
# 2. RMSE
# =========================================================
pivot_rmse = df_summary.pivot_table(index="Model", columns="Treatment", 
                                    values="RMSE", aggfunc='mean')
create_heatmap(pivot_rmse, "RMSE Comparison", "Reds_r")

# =========================================================
# 3. AIC
# =========================================================
pivot_aic = df_summary.pivot_table(index="Model", columns="Treatment", 
                                   values="AIC", aggfunc='mean')
create_heatmap(pivot_aic, "AIC Comparison", "coolwarm")

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
# 9. SAVE OUTPUT
# =========================================================

output_file = r'e:\Document 2026\Hydrogen project\Dark\Biohydrogen from CVH\result_\Step2_Pretreatment.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_summary.to_excel(writer, index=False, sheet_name="Summary")
    df_best.to_excel(writer, index=False, sheet_name="Best_Model")
    perf_df.to_excel(writer, index=False, sheet_name="Ranking")

print("DONE:", output_file)