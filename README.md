# Data Cleaning Challenge: Startup Funding

This folder contains three versions of a startup funding dataset designed to test data preprocessing and cleaning pipelines.

## Task Files

### 1. task_easy.csv
- **Goal:** Basic imputation.
- **Issue:** 5 missing values in the `Revenue_M` column.
- **Action:** Identify and fill missing values using mean or median.

### 2. task_medium.csv
- **Goal:** String standardization and handling sparsity.
- **Issue:** The `Industry` column contains duplicate categories due to casing and typos (e.g., "fintech" vs "Fintech "). Roughly 15% of numerical data is missing.
- **Action:** Clean the categorical strings and handle missing values.

### 3. task_hard.csv
- **Goal:** Anomaly detection and feature selection.
- **Issue:** - **Outliers:** Impossible values in `Employee_Count` (999,999) and negative `Revenue_M`.
    - **Noise:** A completely useless `Noise_Index` column.
    - **Sparsity:** 20% of numerical data is missing.
- **Action:** Drop useless features, clip/remove outliers, and perform robust imputation.
