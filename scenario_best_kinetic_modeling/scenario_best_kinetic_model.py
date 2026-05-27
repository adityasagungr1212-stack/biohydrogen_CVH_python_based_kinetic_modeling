import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 0. STYLE + Q1 CEJ PALETTE (ELEGANT)
# =========================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200
})

scenario_order = ["S1 (<1)", "S2 (~1)", "S3 (1-3.6)", "S4 (>3.6)"]

scenario_palette = {
    "S1 (<1)": "#4C78A8",    # muted blue
    "S2 (~1)": "#F58518",    # soft orange
    "S3 (1-3.6)": "#54A24B", # muted green
    "S4 (>3.6)": "#E45756"   # soft red
}

# =========================================================
# 1. WEIBULL MODEL
# =========================================================

def weibull(t, Hmax, b, n):
    return Hmax * (1 - np.exp(-(b * t) ** n))

# =========================================================
# 2. DATA
# =========================================================

file_path = r"e:\Document 2026\Hydrogen project\Dark\Pyton based kinetic modeling\Step 3.xlsx"

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)

time = df["Time"].values
conditions = df.columns.drop("Time")

# =========================================================
# 3. STORAGE
# =========================================================

fit_results = []
scenario_results = []
weibull_params = {}

# =========================================================
# 4. BASE FIT
# =========================================================

for cond in conditions:

    y_true = df[cond].values
    H0 = np.nanmax(y_true)

    try:
        popt, _ = curve_fit(
            weibull,
            time,
            y_true,
            p0=[H0, 0.1, 1],
            bounds=(0, np.inf),
            maxfev=50000
        )

        y_pred = weibull(time, *popt)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        fit_results.append({
            "Condition": cond,
            "Hmax": popt[0],
            "b": popt[1],
            "n": popt[2],
            "R2": r2,
            "RMSE": rmse
        })

        weibull_params[cond] = {
            "Hmax": popt[0],
            "b": popt[1]
        }

    except Exception as e:
        print(f"FIT FAIL {cond}: {e}")

df_fit = pd.DataFrame(fit_results)

# =========================================================
# 5. SCENARIO SETUP
# =========================================================

scenario_beta = {
    "S1 (<1)": np.linspace(0.2, 0.9, 30),
    "S2 (~1)": np.linspace(0.95, 1.05, 30),
    "S3 (1-3.6)": np.linspace(1.1, 3.6, 50),
    "S4 (>3.6)": np.linspace(3.7, 8.0, 50)
}

# =========================================================
# 6. SCENARIO ANALYSIS (MULTI-OBJECTIVE OPTIMIZATION)
# =========================================================

scenario_results = []

for cond in conditions:

    y_true = df[cond].values
    Hmax = weibull_params[cond]["Hmax"]
    b = weibull_params[cond]["b"]
    n = len(y_true)

    all_beta_results = []

    # store ALL beta evaluations first
    for scenario, betas in scenario_beta.items():

        for beta in betas:

            y_pred = weibull(time, Hmax, b, beta)

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            aic = n * np.log(ss_res / n) + 2 * 3 if ss_res > 0 else np.nan

            all_beta_results.append({
                "Condition": cond,
                "Scenario": scenario,
                "beta": beta,
                "R2": r2,
                "RMSE": rmse,
                "AIC": aic
            })

    temp_df = pd.DataFrame(all_beta_results)

    # =====================================================
    # NORMALIZATION (per condition)
    # =====================================================

    temp_df["R2_norm"] = (temp_df["R2"] - temp_df["R2"].min()) / (temp_df["R2"].max() - temp_df["R2"].min() + 1e-9)

    temp_df["RMSE_norm"] = (temp_df["RMSE"].max() - temp_df["RMSE"]) / (temp_df["RMSE"].max() - temp_df["RMSE"].min() + 1e-9)

    temp_df["AIC_norm"] = (temp_df["AIC"].max() - temp_df["AIC"]) / (temp_df["AIC"].max() - temp_df["AIC"].min() + 1e-9)

    # =====================================================
    # COMPOSITE SCORE
    # =====================================================

    temp_df["Score"] = (
        temp_df["R2_norm"]
        + temp_df["RMSE_norm"]
        + temp_df["AIC_norm"]
    )

    # =====================================================
    # BEST RESULT PER SCENARIO
    # =====================================================

    for scenario in scenario_beta.keys():

        df_s = temp_df[temp_df["Scenario"] == scenario]
        best_row = df_s.loc[df_s["Score"].idxmax()]

        scenario_results.append({
            "Condition": cond,
            "Scenario": scenario,
            "Best_beta": best_row["beta"],
            "R2": best_row["R2"],
            "RMSE": best_row["RMSE"],
            "AIC": best_row["AIC"],
            "Score": best_row["Score"]
        })

scenario_df = pd.DataFrame(scenario_results)

# =========================================================
# BEST SCENARIO PER CONDITION
# =========================================================

best_scenario_df = scenario_df.loc[
    scenario_df.groupby("Condition")["Score"].idxmax()
]

print(best_scenario_df)

# =========================================================
# 7. FIGURE 1 — R²
# =========================================================

plt.figure(figsize=(6,4))
sns.barplot(data=scenario_df, x="Scenario", y="R2", palette=scenario_palette)
plt.title("Weibull Scenario - R²")
plt.xticks(rotation=25)
sns.despine()
plt.tight_layout()
plt.show()

# =========================================================
# 8. FIGURE 2 — RMSE
# =========================================================

plt.figure(figsize=(6,4))
sns.barplot(data=scenario_df, x="Scenario", y="RMSE", palette=scenario_palette)
plt.title("Weibull Scenario - RMSE")
plt.xticks(rotation=25)
sns.despine()
plt.tight_layout()
plt.show()

# =========================================================
# 9. FIGURE 3 — AIC
# =========================================================

plt.figure(figsize=(6,4))
sns.barplot(data=scenario_df, x="Scenario", y="AIC", palette=scenario_palette)
plt.title("Weibull Scenario - AIC")
plt.xticks(rotation=25)
sns.despine()
plt.tight_layout()
plt.show()

# =========================================================
# 10. FIGURE 4 — OPTIMAL β
# =========================================================

best_beta_df = scenario_df.loc[
    scenario_df.groupby("Condition")["R2"].idxmax()
]

plt.figure(figsize=(7,4))
sns.barplot(data=best_beta_df, x="Condition", y="Best_beta", palette="Blues")
plt.axhline(1, linestyle="--", color="black", linewidth=1)
plt.axhline(3.6, linestyle=":", color="gray", linewidth=1)
plt.title("Optimal Weibull β per Condition")
plt.xticks(rotation=45)
sns.despine()
plt.tight_layout()
plt.show()

# =========================================================
# 13. CURVE FIT VISUALIZATION (BOXED + BOLD STYLE)
# =========================================================

for cond in conditions:

    y_true = df[cond].values
    Hmax = weibull_params[cond]["Hmax"]
    b = weibull_params[cond]["b"]

    plt.figure(figsize=(7,5))

    plt.scatter(time, y_true, color="black", s=25, label="Experimental")

    for scenario in scenario_beta.keys():

        beta = scenario_beta[scenario].mean()
        y_pred = weibull(time, Hmax, b, beta)

        plt.plot(
            time,
            y_pred,
            color=scenario_palette[scenario],
            linewidth=2,
            label=scenario
        )

    # =====================================================
    # LABELS + TITLE (BOLD)
    # =====================================================
    plt.title(f"Weibull Hydrogen Yield - {cond}", fontweight="bold")
    plt.xlabel("Time (h)", fontweight="bold", fontsize=10)
    plt.ylabel("Cumulative hydrogen yield (mL H$_2$/g VS)", fontweight="bold", fontsize=10)

    # =====================================================
    # LEGEND (BOLD)
    # =====================================================
    plt.legend(frameon=True)

    # =====================================================
# FULL BOX AXES (TOP + RIGHT ENABLED)
# =====================================================

ax = plt.gca()

# show all spines (THIS IS THE KEY)
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)

# thickness biar tegas (Q1 style)
ax.spines["top"].set_linewidth(1.2)
ax.spines["right"].set_linewidth(1.2)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

# tick styling
ax.tick_params(axis='both', which='both', width=1.2, labelsize=10)
plt.tight_layout()
plt.show()

# =========================================================
# 11. EXPERIMENTAL vs PREDICTED (ALL SCENARIOS)
# =========================================================

exp_pred_all = []

for cond in conditions:

    y_true = df[cond].values
    Hmax = weibull_params[cond]["Hmax"]
    b = weibull_params[cond]["b"]

    for scenario, betas in scenario_beta.items():

        beta_opt = scenario_df[
            (scenario_df["Condition"] == cond) &
            (scenario_df["Scenario"] == scenario)
        ]["Best_beta"].values[0]

        y_pred = weibull(time, Hmax, b, beta_opt)

        for t, yt, yp in zip(time, y_true, y_pred):

            exp_pred_all.append({
                "Condition": cond,
                "Time": t,
                "Scenario": scenario,
                "Beta": beta_opt,
                "Experimental": yt,
                "Predicted": yp
            })

df_exp_pred = pd.DataFrame(exp_pred_all)

# =========================================================
# 14. SAVE OUTPUT
# =========================================================

output_file = r"e:\Document 2026\Hydrogen project\Dark\Pyton based kinetic modeling\Weibull_FINAL.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_fit.to_excel(writer, sheet_name="Fit", index=False)
    scenario_df.to_excel(writer, sheet_name="Scenario", index=False)
    df_exp_pred.to_excel(writer, sheet_name="Exp_vs_Pred_Scenario", index=False)

print("DONE:", output_file)