# Python-Based Kinetic Modeling and Scenario Analysis for Biohydrogen Production from C. vulgaris FSP-E hydrolysate (CVH)

## Overview
This repository contains experimental datasets, FTIR characterization data, Python-based kinetic modeling, and scenario analysis for biohydrogen production from *Chlorella vulgaris* hydrolysate under different pretreatment conditions.

The study integrates:
- Multi-step experimental data (Step 1–3)
- FTIR characterization of different pretreatments
- Kinetic modeling using nonlinear regression (curve fitting)
- Model comparison (Gompertz, First-order, Richards, Weibull)
- Performance evaluation (R², RMSE, AIC)
- Scenario-based interpretation of Weibull β parameter
- TOPSIS-based multi-criteria decision analysis

---

## Repository Structure

### 📁 data_
Contains all raw experimental and processed datasets:

- Step1_Concentration_of_CVH  
- Step1_TOPSIS  
- Step2_Pretreatment  
- Step2_TOPSIS  
- Step3_Chemical_additives  
- Step3_TOPSIS  

This folder is the main input for kinetic modeling and decision analysis.

---

### 📁 FTIR data
Contains FTIR spectral data used to characterize structural differences between pretreatment conditions:

- FTIR data different pretreatment

Used for supporting chemical/structural interpretation of biomass changes.

---

### 📁 python_based_kinetic_modeling
Contains Python scripts for kinetic model fitting and evaluation:

- Step1_Concentration_of_CVH
- Step1_TOPSIS
- Step2_Pretreatment
- Step2_TOPSIS
- Step3_Chemical_additives
- Step3_TOPSIS

Each script performs:
- Nonlinear curve fitting (`scipy.curve_fit`)
- Model comparison across four kinetic models
- Calculation of R², adjusted R², RMSE, and AIC
- Visualization of experimental vs predicted curves
- Export of results to Excel format

---

### 📁 scenario_best_kinetic_modeling
Contains final processed outputs from Python analysis:

- Best-performing kinetic model per condition
- Scenario classification based on Weibull β parameter
- Summary tables and interpreted results

👉 This folder represents the **final scientific interpretation layer**.

---

### 📁 result_
Contains final output files generated from Python scripts:

- Model summary (R², RMSE, AIC)
- Best model selection per condition
- Model ranking results

---

### 📁Scenario Analysis (Weibull β)

The Weibull shape parameter (β) is used to describe hydrogen production behavior:

- β < 1 → decelerating production (early saturation)
- β ≈ 1 → exponential-like growth
- β > 1 → accelerating production phase

This enables classification of kinetic regimes across pretreatment conditions.



---



## TOPSIS Analysis

TOPSIS was applied to:

- Rank pretreatment conditions
- Evaluate multi-criteria performance (H₂ yield, rate, efficiency metrics)
- Support decision-making in selecting optimal conditions

---

## Software Requirements

Python ≥ 3.8

### Required libraries:
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- scikit-learn
- openpyxl

Install dependencies:

```bash
pip install -r requirements.txt