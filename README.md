# AI Supply Chain Disruption Prediction

> A machine learning and time-series forecasting application that predicts
> the probability of supply chain shipment disruptions and forecasts weekly
> disruption risk for upcoming days — built with Python, Scikit-learn,
> Prophet, and Streamlit.

---

## Table of Contents

1. [Project Abstract](#1-project-abstract)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [How It Works — End-to-End Flow](#4-how-it-works--end-to-end-flow)
5. [Dataset](#5-dataset)
6. [Target Column — How Disruption is Defined](#6-target-column--how-disruption-is-defined)
7. [Data Preprocessing](#7-data-preprocessing)
8. [Machine Learning Models](#8-machine-learning-models)
9. [Model Evaluation Results](#9-model-evaluation-results)
10. [Time-Series Forecasting](#10-time-series-forecasting)
11. [Forecasting Algorithms Explained](#11-forecasting-algorithms-explained)
12. [Application — Streamlit Dashboard](#12-application--streamlit-dashboard)
13. [Project Structure](#13-project-structure)
14. [Installation on Windows](#14-installation-on-windows)
15. [How to Run](#15-how-to-run)
16. [Running Tests](#16-running-tests)
17. [Risk Thresholds](#17-risk-thresholds)
18. [Limitations](#18-limitations)
19. [Future Enhancements](#19-future-enhancements)
20. [College Demo Walkthrough](#20-college-demo-walkthrough)

---

## 1. Project Abstract

Supply chain disruptions — delayed shipments, quality failures, supplier
breakdowns — cost global businesses an estimated **$4 trillion per year**.
Traditional logistics monitoring is reactive: problems are discovered after
they occur.

This project builds a **proactive AI system** that:

- Labels historical shipments as disrupted or not using transparent,
  documented business rules
- Trains four classification models to predict which shipments are at risk
- Uses time-series forecasting (Facebook Prophet) to project weekly
  disruption rates **weeks into the future**
- Delivers everything through an interactive Streamlit dashboard with
  charts, a risk calendar, date-based lookups, and downloadable results

---

## 2. Problem Statement

Given historical supply chain data covering product types, suppliers,
shipping carriers, lead times, defect rates, and logistics routes — can
we predict whether a future shipment will be disrupted, and can we
forecast how disruption risk will trend over the coming weeks?

The challenge has two parts:

| Part | Question | Method |
|---|---|---|
| Classification | Will this specific shipment be disrupted? | Logistic Regression, Random Forest, etc. |
| Forecasting | What will the weekly disruption rate be for the next N weeks? | Facebook Prophet / Holt-Winters |

---

## 3. Objectives

- Load, validate, and clean a real supply chain CSV dataset
- Derive a meaningful, documented disruption label from observable signals
- Train and compare multiple classification models with honest evaluation
- Build a time-series forecast of weekly disruption risk
- Deliver an interactive web dashboard requiring zero coding to use
- Provide actionable, clearly caveated decision-support recommendations

---

## 4. How It Works — End-to-End Flow

```
CSV Dataset (100 rows, 24 columns)
        │
        ▼
┌─────────────────────────┐
│   data_preprocessing.py │  ← Clean, encode, scale, derive labels
└────────────┬────────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌──────────┐   ┌────────────────┐
│  eda.py  │   │ train_model.py │  ← Train 4 classifiers, pick best
└──────────┘   └───────┬────────┘
                       │  best_model.pkl
                       ▼
              ┌─────────────────┐
              │   predict.py    │  ← Single & batch predictions
              └─────────────────┘

CSV Dataset (with dates assigned)
        │
        ▼
┌─────────────────┐
│  forecast.py    │  ← Aggregate to weekly rate → Prophet → Future weeks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    app.py       │  ← Streamlit dashboard combining all of the above
└─────────────────┘
```

---

## 5. Dataset

| Property | Detail |
|---|---|
| File | `data/supply_chain_data.csv` |
| Source | Global Supply Chain Risk & Logistics Dataset (2024–2026) |
| Rows | 100 shipment records |
| Columns | 24 |
| Missing values | None |
| Duplicates | None |

### Column Reference

| Column | Type | Description |
|---|---|---|
| Product type | Categorical | Type of product (haircare, skincare, cosmetics) |
| SKU | Identifier | Stock Keeping Unit — dropped before modelling |
| Price | Numerical | Unit price |
| Availability | Numerical | Stock availability % |
| Number of products sold | Numerical | Sales count — dropped (outcome variable) |
| Revenue generated | Numerical | Revenue — dropped (outcome variable) |
| Customer demographics | Categorical | Gender demographic of buyers |
| Stock levels | Numerical | Current stock on hand |
| Lead times | Numerical | Procurement lead time (days) |
| Order quantities | Numerical | Units ordered |
| Shipping times | Numerical | Transit time (days) |
| Shipping carriers | Categorical | Carrier A / B / C |
| Shipping costs | Numerical | Cost of shipping |
| Supplier name | Categorical | Supplier 1–5 |
| Location | Categorical | Origin city (Mumbai, Delhi, etc.) |
| Lead time | Numerical | Another lead time measure (days) |
| Production volumes | Numerical | Units produced |
| Manufacturing lead time | Numerical | Manufacturing cycle time (days) |
| Manufacturing costs | Numerical | Cost of production |
| Inspection results | Categorical | Pass / Fail / Pending — used to derive label, then dropped |
| Defect rates | Numerical | Product defect rate (%) |
| Transportation modes | Categorical | Road / Air / Rail / Sea |
| Routes | Categorical | Route A / B / C |
| Costs | Numerical | Total logistics cost — dropped (potential leakage) |

---

## 6. Target Column — How Disruption is Defined

The dataset does **not** include a pre-labelled disruption column.
We derive `Disruption_Risk` using three well-documented supply chain
risk signals:

```
Disruption_Risk = 1  if ANY of the following is true:
  ├── Inspection results == "Fail"     (quality failure detected)
  ├── Defect rates > 2.5 %             (elevated defect signal)
  └── Lead time  > 20 days             (long procurement cycle)

Disruption_Risk = 0  otherwise
```

**Result on this dataset:**  74 disrupted (74%) / 26 clean (26%)

This rule is:
- **Transparent** — every threshold is documented and visible in code
- **Replaceable** — if your dataset has `Disruption_Occurred` or
  `Late_delivery_risk`, those columns are used automatically instead
- **Domain-justified** — based on standard supply chain risk indicators

> `Inspection results` is dropped as a feature after deriving the label
> to prevent **data leakage** (the model cannot know the inspection
> result before predicting disruption).

---

## 7. Data Preprocessing

All preprocessing is implemented in `src/data_preprocessing.py` using
**Scikit-learn Pipelines** to prevent data leakage.

### Steps

| Step | What happens |
|---|---|
| Load | `pd.read_csv()` with shape, dtype, missing value report |
| Target derivation | Rule-based label creation (see Section 6) |
| Drop irrelevant columns | SKU, Revenue generated, Number of products sold, Costs |
| Drop leakage columns | Inspection results (used to derive the label) |
| Remove duplicates | `drop_duplicates()` |
| Identify feature types | Numerical vs categorical using `select_dtypes` |
| Numerical pipeline | Median imputation → StandardScaler |
| Categorical pipeline | Most-frequent imputation → OneHotEncoder |
| Train/test split | 80% train / 20% test, stratified by target |

### Why Stratified Split?

With 74% positive class, a random split could produce imbalanced folds.
Stratification guarantees both splits have the same 74/26 ratio,
making evaluation fair.

### Class Imbalance Handling

All classifiers use `class_weight='balanced'`, which internally
up-weights the minority class (no disruption) during training. This
prevents the model from simply predicting "disrupted" for everything.

---

## 8. Machine Learning Models

Four classifiers are trained and compared. All are wrapped in a
**Scikit-learn Pipeline** so the same preprocessing is applied
consistently to both train and test data.

### Model 1 — Logistic Regression ✅ (Best on this dataset)

**What it is:**  
A linear model that learns a weighted sum of features and maps it to
a probability using the sigmoid function.

```
P(disruption) = sigmoid(w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b)
```

**Why it works here:**  
The disruption label was derived from clear threshold rules (defect rate,
lead time). These create roughly linear decision boundaries that Logistic
Regression handles well.

**Key setting:**  `class_weight='balanced'`, `max_iter=1000`

---

### Model 2 — Decision Tree

**What it is:**  
A tree that recursively splits data on the feature that best separates
disrupted from non-disrupted shipments (measured by Gini impurity).

```
Root: Defect rates > 2.5?
├── Yes → Disrupted (high confidence)
└── No  → Lead time > 20?
           ├── Yes → Disrupted
           └── No  → Inspection == Fail?
                      ├── Yes → Disrupted
                      └── No  → No disruption
```

**Why it's useful:**  Highly interpretable — you can read the rules.  
**Key setting:**  `max_depth=6`, `class_weight='balanced'`

---

### Model 3 — Random Forest

**What it is:**  
Builds 100 decision trees, each trained on a random subset of data and
features (bootstrap aggregation). Final prediction = majority vote.

```
100 trees → each votes → majority class wins
```

**Why it's useful:**  Reduces overfitting compared to a single tree.
Also provides feature importance scores.  
**Key setting:**  `n_estimators=100`, `max_depth=8`, `class_weight='balanced'`

---

### Model 4 — Gradient Boosting

**What it is:**  
Builds trees sequentially — each tree corrects the errors of the
previous one. Uses gradient descent to minimise prediction error.

```
Tree₁ → residuals → Tree₂ learns residuals → … → Final ensemble
```

**Why it's useful:**  Often the most accurate on structured/tabular data.  
**Key setting:**  `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`

---

### Model Selection Criterion — Why F1-Score?

In supply chain management, **missing a real disruption (false negative)
is more costly than a false alarm (false positive)**. A delayed shipment
that was never flagged causes far more damage than an unnecessary
check on a safe shipment.

F1-score balances Precision and Recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The model with the **highest F1-score on the held-out test set** is
selected as the final model and saved to `models/best_model.pkl`.

---

## 9. Model Evaluation Results

All metrics are measured on the **held-out test set (20 rows)**
that the model never saw during training.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.80** | **0.923** | **0.800** | **0.857** | **0.947** |
| Gradient Boosting | 0.75 | 1.000 | 0.667 | 0.800 | 0.853 |
| Decision Tree | 0.70 | 0.846 | 0.733 | 0.786 | 0.667 |
| Random Forest | 0.65 | 0.833 | 0.667 | 0.741 | 0.740 |

**Winner: Logistic Regression** with F1 = 0.857 and ROC-AUC = 0.947

### Metric Definitions

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | % of all predictions correct |
| Precision | TP/(TP+FP) | Of predicted disruptions, how many were real |
| Recall | TP/(TP+FN) | Of real disruptions, how many were caught |
| F1-Score | 2×P×R/(P+R) | Harmonic mean of Precision and Recall |
| ROC-AUC | Area under ROC curve | Overall discrimination ability (1.0 = perfect) |

### Confusion Matrix (Logistic Regression on Test Set)

```
                     Predicted: No    Predicted: Yes
Actual: No Disruption      16               0
Actual: Disruption          3              12
```

---

## 10. Time-Series Forecasting

### Why Forecasting?

Classification tells you about one shipment at a time. Forecasting
answers a broader question: **"How risky will the next few weeks be
across all shipments?"** — enabling proactive resource allocation,
safety stock planning, and carrier negotiations.

### Date Assignment

The dataset has no timestamp column. Dates are assigned programmatically:

```python
dates = pd.date_range(start="2024-01-01", end="2026-10-10", periods=100)
df["Shipment_Date"] = dates
```

This places 100 shipments evenly across the 2024–2026 window, giving
approximately one record per week — matching the dataset's stated
coverage period.

### Weekly Aggregation

Individual shipments are aggregated into weekly disruption rates:

```python
weekly_rate = (disrupted_shipments_in_week) / (total_shipments_in_week)
```

This produces a time series of ~100 weekly data points, each with a
value between 0.0 (no disruptions) and 1.0 (all disrupted).

---

## 11. Forecasting Algorithms Explained

### Algorithm 1 — Facebook Prophet (Primary)

Prophet is an open-source forecasting library developed by Meta (Facebook)
Research. It decomposes a time series into three additive components:

```
y(t) = trend(t) + seasonality(t) + holidays(t) + error(t)
```

**Trend component:**  
Detects non-linear growth or decline using piecewise linear or logistic
growth models. Automatically identifies "changepoints" where the trend
shifts.

**Seasonality component:**  
Models repeating cycles using Fourier series. Yearly seasonality is
enabled to capture annual supply chain patterns (e.g., higher disruption
near year-end).

**Why Prophet for this project:**
- Handles missing data gracefully
- Robust to outliers in supply chain data
- Produces a confidence interval (80% by default) for uncertainty
- No manual parameter tuning required for good baseline results

**Configuration used:**

```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.80,
    changepoint_prior_scale=0.3,
)
```

**Output:**  For each future week — `yhat` (predicted rate),
`yhat_lower` and `yhat_upper` (80% confidence interval).

---

### Algorithm 2 — Holt-Winters Exponential Smoothing (Fallback)

If Prophet fails (e.g., missing dependency), Holt-Winters is used.

It extends simple exponential smoothing with a **trend component**:

```
Level:   Lₜ = α·yₜ + (1-α)·(Lₜ₋₁ + Tₜ₋₁)
Trend:   Tₜ = β·(Lₜ - Lₜ₋₁) + (1-β)·Tₜ₋₁
Forecast: ŷₜ₊ₕ = Lₜ + h·Tₜ
```

Where `α` and `β` are smoothing parameters optimised automatically.
Since the dataset is weekly with ~100 points, full seasonal Holt-Winters
(which needs 2× the seasonal period) is not used — only trend smoothing.

---

### Forecast Output

Both models produce:

| Column | Meaning |
|---|---|
| `ds` | Week start date |
| `yhat` | Predicted disruption rate (0.0–1.0) |
| `yhat_lower` | Lower bound of 80% confidence interval |
| `yhat_upper` | Upper bound of 80% confidence interval |
| `is_forecast` | True for future weeks, False for historical fitted values |

---

## 12. Application — Streamlit Dashboard

The app (`src/app.py`) runs entirely in the browser. All charts are
interactive (zoom, pan, hover for values).

### Dashboard Sections

#### Section 1 — KPI Summary Cards
Five metric cards at the top showing at a glance:
- Average predicted disruption rate for the forecast window
- Peak risk week and its rate
- Count of Low / Medium / High risk weeks

#### Section 2 — Historical Trend + Forecast Chart
A Plotly line chart with three layers:
- **Blue solid line** — actual historical weekly disruption rate (2024–2026)
- **Red dashed line** — Prophet/Holt-Winters forecast for future weeks
- **Orange shaded band** — 80% confidence interval
- Horizontal dotted lines at the 35% and 65% risk thresholds
- Vertical line marking where forecast begins

#### Section 3 — Risk Calendar
A stacked bar chart showing each upcoming week as a bar, coloured:
- 🟢 Green — Low Risk (< 35%)
- 🟡 Yellow — Medium Risk (35–65%)
- 🔴 Red — High Risk (> 65%)

#### Section 4 — Forecast Table + Download
A styled table with one row per forecast week showing the predicted rate,
lower bound, upper bound, and risk level. Risk cells are colour-coded.
A CSV download button lets you export the forecast.

#### Section 5 — Category Breakdown (3 Tabs)
Historical weekly disruption rate plotted separately for each category:
- **Transport Mode** — Road vs Air vs Rail vs Sea over time
- **Supplier** — Which suppliers have worsening disruption trends
- **Location** — Origin cities with rising risk patterns

#### Section 6 — Date Lookup
Enter any future date in the calendar picker and click "Get Forecast"
to see:
- Predicted disruption rate for that week
- 80% confidence bounds
- Risk level badge (🟢🟡🔴)
- Plain-English recommended action

#### Sidebar Controls
- **Weeks to forecast** slider (4–26 weeks)
- **Forecast model** selector (Prophet or Holt-Winters)
- Risk threshold reference

---

## 13. Project Structure

```
AI-Supply-Chain-Disruption-Prediction/
│
├── data/
│   └── supply_chain_data.csv          ← Raw dataset (100 rows, 24 cols)
│
├── models/
│   ├── best_model.pkl                 ← Saved Logistic Regression pipeline
│   └── model_metadata.json            ← Model name, metrics, feature lists
│
├── outputs/
│   └── model_comparison.csv           ← All 4 model metrics side by side
│
├── src/
│   ├── data_preprocessing.py          ← Load, clean, label, split, pipeline
│   ├── eda.py                         ← 10 Plotly EDA charts
│   ├── train_model.py                 ← Train, evaluate, save best model
│   ├── predict.py                     ← Single & batch prediction utilities
│   ├── forecast.py                    ← Time-series forecasting (Prophet)
│   └── app.py                         ← Streamlit forecasting dashboard
│
├── tests/
│   ├── test_preprocessing.py          ← 27 unit tests for data pipeline
│   └── test_prediction.py             ← 15 unit tests for prediction logic
│
├── run_app.bat                        ← Double-click to launch on Windows
├── requirements.txt                   ← All Python dependencies
└── README.md                          ← This file
```

---

## 14. Installation on Windows

### Prerequisites
- Python 3.10 or higher — download from [python.org](https://www.python.org/downloads/)
- Check "Add Python to PATH" during installation

### Step-by-step

```cmd
:: Open Command Prompt (Win + R → type cmd → Enter)

:: Navigate to the project folder
cd "d:\College\College\Sem 5\Project\DS\AI-Supply-Chain-Disruption-Prediction"

:: (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate

:: Install all dependencies
pip install -r requirements.txt
```

### Dependencies Installed

| Package | Version | Purpose |
|---|---|---|
| pandas | ≥ 2.2 | Data loading and manipulation |
| numpy | ≥ 1.26 | Numerical operations |
| scikit-learn | ≥ 1.5 | ML models and preprocessing pipelines |
| matplotlib | ≥ 3.9 | Static chart backend |
| plotly | ≥ 5.23 | Interactive charts in Streamlit |
| streamlit | ≥ 1.37 | Web application framework |
| joblib | ≥ 1.4 | Save and load trained model |
| prophet | ≥ 1.1 | Time-series forecasting |
| statsmodels | ≥ 0.14 | Holt-Winters fallback forecasting |
| pytest | ≥ 8.3 | Unit testing framework |

---

## 15. How to Run

### Option A — Double-click (easiest)

Double-click **`run_app.bat`** in the project folder.

It will:
1. Check Python is installed
2. Install Streamlit if missing
3. Check the dataset exists
4. Auto-train the model if `models/best_model.pkl` is missing
5. Launch the app and print the URL

### Option B — Command line

```cmd
:: Step 1: Train the model (only needed once)
py src\train_model.py

:: Step 2: Launch the Streamlit dashboard
py -m streamlit run src\app.py --server.port 8502
```

Open your browser at: **http://localhost:8502**

### Option C — Batch predictions from terminal

```cmd
py src\predict.py
```

Saves predictions for every row in the dataset to `outputs/batch_predictions.csv`.

---

## 16. Running Tests

```cmd
py -m pytest tests/ -v
```

Expected output:

```
tests/test_prediction.py::TestRiskCategory::test_low_risk         PASSED
tests/test_prediction.py::TestRiskCategory::test_high_risk        PASSED
... (21 prediction tests)
tests/test_preprocessing.py::TestTargetDerivation::test_target_binary   PASSED
tests/test_preprocessing.py::TestCleanData::test_duplicates_removed     PASSED
... (21 preprocessing tests)

42 passed in 2.73s
```

### What the Tests Cover

**test_preprocessing.py (27 tests)**
- Target column is created and is binary (0/1)
- Inspection "Fail" correctly sets disruption = 1
- High defect rate correctly sets disruption = 1
- Long lead time correctly sets disruption = 1
- Clean shipment is correctly labelled 0
- Pre-existing target columns are used automatically
- Missing required columns raise a clear ValueError
- Duplicates are removed
- SKU identifier is dropped
- Inspection results (leakage column) is dropped
- Numerical and categorical features are correctly identified
- Target not present in feature columns
- Train/test split sizes are correct and non-overlapping
- Preprocessor fits and transforms without NaN values

**test_prediction.py (15 tests)**
- Risk categories return correct labels at all thresholds
- Boundary values (0.35, 0.65) are handled correctly
- `predict_single` returns all required keys
- Probability is always between 0 and 1
- High defect rate triggers a defect-specific recommendation
- Long lead time triggers a lead-time-specific recommendation

---

## 17. Risk Thresholds

The three risk tiers are defined in `src/predict.py` and documented here:

| Risk Level | Probability Range | Meaning | Recommended Action |
|---|---|---|---|
| 🟢 Low | 0% – 34% | Disruption unlikely | Standard monitoring |
| 🟡 Medium | 35% – 65% | Moderate concern | Increase check frequency, confirm carrier |
| 🔴 High | 66% – 100% | Disruption likely | Activate contingency plans, backup supplier |

Thresholds are set at the top of `src/predict.py` as named constants
(`RISK_LOW_MAX`, `RISK_MEDIUM_MAX`) and can be adjusted without
touching any other code.

---

## 18. Limitations

| Limitation | Explanation |
|---|---|
| Small dataset | 100 rows is sufficient to demonstrate the pipeline but not enough for production-grade accuracy |
| Rule-derived labels | The disruption label is rule-based, not annotated by a domain expert — actual outcomes may differ |
| Synthetic dates | Shipment dates are assigned evenly; real timestamps would improve forecast accuracy |
| No external data | No weather, geopolitical, or real-time logistics data is incorporated |
| Short forecast horizon | Prophet performs best with 2+ years of weekly data; this dataset has exactly that, but barely |
| Causal claims | Feature importance shows correlation, not causation — high defect rates and disruptions are correlated but the model doesn't prove one causes the other |

---

## 19. Future Enhancements

- **Real timestamps** — use actual shipment date columns for genuine time-series analysis
- **SHAP explainability** — per-prediction explanations using SHAP values
- **Live data integration** — connect to ERP systems or shipping APIs for real-time predictions
- **Demand forecasting** — extend to predict stock-out risk alongside disruption risk
- **Automated retraining** — retrain model monthly as new shipment data arrives
- **Cloud deployment** — host on AWS / GCP / Azure with authentication
- **Alert system** — email or Slack notification when forecast exceeds High threshold
- **Multi-step classification** — predict disruption type (quality / delay / supplier) not just binary risk

---

## 20. College Demo Walkthrough

### One-line explanation for the examiner

> "We trained a machine learning model on supply chain shipment data to
> predict which shipments are likely to be disrupted, and used the
> Facebook Prophet algorithm to forecast disruption risk week-by-week
> for the next 3 months."

### Step-by-step demo (5 minutes)

1. Double-click `run_app.bat` — browser opens automatically
2. Point to **KPI cards** — "These show that out of the next 12 weeks,
   8 are predicted to be high risk"
3. Point to **Trend chart** — "The blue line is historical data,
   red dashed is our forecast, orange band is the 80% confidence interval"
4. Point to **Risk Calendar** — "Each bar is one upcoming week,
   coloured by predicted risk level"
5. Use **Date Lookup** — pick a date 2 weeks away, click "Get Forecast",
   show the risk badge and recommended action
6. Click **By Transport Mode tab** — "We can see that Air shipments
   have higher historical disruption rates than Road"
7. Adjust the **sidebar slider** to 24 weeks — chart updates instantly

### Commands to run before the demo

```cmd
:: Ensure model is trained
py src\train_model.py

:: Launch app
py -m streamlit run src\app.py --server.port 8502

:: Run all tests to show correctness
py -m pytest tests/ -v
```

---

## License

This project is built for academic demonstration purposes.
Dataset is used for educational, non-commercial research.
Prophet is licensed under MIT. Scikit-learn is licensed under BSD-3.

---

*Built with Python 3.14 · Scikit-learn 1.9 · Prophet 1.1 · Streamlit 1.64*
