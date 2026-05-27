# Cal Poly Rec Center Usage Prediction — Final Report

**Advanced Machine Learning Final Project**

---

## 1. Problem Statement

The Cal Poly Recreation Center is one of the most heavily used student facilities on campus, yet students often have no reliable way to know how crowded it will be before they arrive. Long lines, full weight rooms, and packed cardio areas create frustration and can discourage students from using a resource they already pay for.

This project addresses a practical question: **Can we predict how busy the Rec Center (and specific areas within it) will be at a given date and time?**

We built two complementary prediction tools:

1. **Regression model** — estimates exact utilization (occupancy relative to capacity, on a 0–1 scale).
2. **Classification model** — labels each time slot as **low**, **medium**, or **high** usage for easy student-facing recommendations.

These predictions can support student planning ("When is the best time to go?"), Rec Center staffing decisions, and future integration with a mobile app or website.

---

## 2. Data Description

### Source and coverage

The data comes from **Occuspace**, a occupancy-analytics platform that estimates how many people are in tracked spaces. We used Cal Poly Rec Center exports collected every **30 minutes** from **6:00 AM to 11:30 PM**, spanning **May 2023 through April 2026**.

After cleaning, the modeling dataset contains **223,283 observations** across **six locations**:

| Location | Role |
|---|---|
| Rec Center (overall) | Whole-facility crowding |
| 1st Floor | General floor traffic |
| 2nd Floor | General floor traffic |
| Lower Exercise Room | Strength/cardio area |
| Upper Exercise Room | Strength/cardio area |
| Track Exercise Room | Track and adjacent equipment |

### Key variables

- **Average Utilization** (primary target): average occupancy divided by listed capacity. A value of 0.50 means the space is about half full; 1.0 means at capacity.
- **Time predictors**: hour of day, day of week, month, weekend indicator.
- **Location**: which area of the Rec Center is being measured.
- **Capacity**: maximum listed occupancy for each space (used for context, not as a standalone predictor for utilization).

### Data quality notes

- No missing values in the original export columns.
- **216 duplicate** location/timestamp pairs were removed.
- **22,069 rows** had average utilization above 1.0 (Occuspace estimated more people than listed capacity). We retained these observations but **capped utilization at 1.0** for modeling, since values above 100% likely reflect sensor or capacity-listing error rather than a fundamentally different crowding pattern.

---

## 3. Exploratory Analysis

Our analysis of three years of usage data revealed clear, actionable patterns.

### Overall utilization

Average utilization across all locations and times is **0.44** (44% of capacity), with a median of **0.35**. The distribution is right-skewed: most intervals are moderately busy, but a meaningful share of periods are very crowded.

![Distribution of Average Utilization](../figures/average_utilization_distribution.png)

### Time-of-day patterns

Usage is lowest in the early morning and late evening. Crowding builds through the afternoon and peaks between **4:00 PM and 6:00 PM**, with **5:00 PM** showing the highest average utilization.

![Average Utilization by Hour](../figures/utilization_by_hour.png)

### Day-of-week patterns

**Monday and Tuesday** are the busiest days, followed by Wednesday. **Saturday and Sunday** are consistently quieter — roughly 15–20% lower average utilization than midweek days.

![Average Utilization by Weekday](../figures/utilization_by_weekday.png)

### Location differences

Not all areas are equally crowded. The **Track Exercise Room** averages **0.58** utilization, followed by the **2nd Floor** (0.52) and **Upper Exercise Room** (0.49). The **1st Floor** and **Lower Exercise Room** are the least crowded at roughly **0.31** each — useful alternatives when popular areas are full.

![Average Utilization by Location](../figures/utilization_by_location.png)

### Seasonal and academic calendar effects

Monthly trends show visible dips during **summer** and **winter break**, and higher usage during active academic quarters. August 2023 and summer months consistently show lower utilization, while May 2024 showed the highest monthly average — likely reflecting pre-finals and end-of-quarter activity.

![Monthly Utilization Trend](../figures/monthly_utilization_trend.png)

![Utilization by Academic Period](../figures/utilization_by_academic_period.png)

### Weekday–hour interactions

A heatmap of weekday versus hour confirms that **weekday late afternoons** are the hottest windows, while **weekend mornings** and **late evenings** are consistently quieter. These interaction patterns motivated our use of flexible, nonlinear models rather than simple linear formulas.

![Weekday-Hour Heatmap](../figures/weekday_hour_heatmap.png)

---

## 4. Modeling Approach

### How we trained and tested

We treated this as a **forecasting problem**: the model learns from past usage and predicts future periods it has never seen. Data was split by time:

| Split | Period | Purpose |
|---|---|---|
| Training | May 2023 – June 2024 | Learn historical patterns |
| Validation | July – December 2024 | Tune model settings |
| Test | January 2025 – April 2026 | Final unbiased evaluation |

We deliberately avoided random train/test splits, which would mix future data into training and overstate accuracy.

### Features used

Beyond raw time and location fields, we engineered:

- **Weekend indicator** and **cyclical time encodings** (sin/cos transforms for hour and month)
- **Academic calendar flags**: summer, winter break, spring break, and finals week (derived from Cal Poly's academic calendar)

We excluded peak occupancy and peak utilization from predictors because they describe the same 30-minute window as the target and would inflate performance artificially.

### Models compared

**Regression (predict exact utilization):**

| Model | Description |
|---|---|
| Historical mean baseline | Average utilization by location × weekday × hour |
| Ridge regression | Linear model with regularization |
| Random Forest | Ensemble of decision trees |
| LightGBM / CatBoost | Gradient boosting (state-of-the-art for tabular data) |
| Keras MLP | Neural network with two hidden layers |
| Stacking ensemble | Combines LightGBM, CatBoost, and MLP via a meta-learner |

**Classification (predict low / medium / high):**

| Category | Threshold |
|---|---|
| Low | Utilization below 0.30 |
| Medium | 0.30 – 0.60 |
| High | Above 0.60 |

These cutoffs align with the data median (0.35) and mean (0.44), giving intuitive "quiet / moderate / crowded" labels.

Models compared: logistic regression, random forest, LightGBM, CatBoost, and a Keras neural network classifier.

All models were tuned on the validation period before final test evaluation.

---

## 5. Results and Model Comparison

### Regression results (test set)

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| **Stacking ensemble** | **0.123** | **0.089** | **0.872** |
| CatBoost | 0.125 | 0.088 | 0.867 |
| LightGBM | 0.128 | 0.088 | 0.862 |
| Random Forest | 0.134 | 0.090 | 0.848 |
| Keras MLP (neural network) | 0.143 | 0.100 | 0.826 |
| Ridge regression | 0.219 | 0.175 | 0.593 |
| Historical mean baseline | 0.289 | 0.247 | 0.291 |

**How to read these metrics:**

- **RMSE (root mean squared error)** — our primary metric. It penalizes large misses heavily, which matters most during peak hours when students care most about accuracy. An RMSE of 0.123 means typical prediction errors are roughly 12 percentage points of utilization.
- **MAE (mean absolute error)** — easier to interpret: on average, predictions are off by about **9 percentage points** for the best models.
- **R²** — share of variance explained. Our best model explains about **87%** of utilization variation on unseen future data.

### Neural network vs. non-neural network

Tree-based models (LightGBM, CatBoost, Random Forest) outperformed the neural network on this dataset. This is a common finding for structured tabular data with strong categorical interactions (location × hour × weekday). The neural network still beat linear baselines by a wide margin but did not justify its added complexity over gradient boosting alone.

The stacking ensemble achieved a modest improvement over the best individual model (RMSE 0.123 vs. 0.125 for CatBoost alone), suggesting some benefit from combining diverse model types, though the gain is small relative to the engineering effort.

![Peak Hour Predictions](../figures/peak_hour_pred_vs_actual.png)

### Classification results (test set)

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| **CatBoost** | **81.9%** | **0.793** |
| LightGBM | 81.3% | 0.787 |
| Random Forest | 81.3% | 0.784 |
| Keras MLP | 80.4% | 0.779 |
| Logistic regression | 70.9% | 0.658 |

**Macro-F1** is our primary classification metric because it weights low, medium, and high usage classes equally — important since high-crowd periods are less frequent but most valuable to predict correctly.

CatBoost correctly classifies usage level about **82%** of the time on held-out future data, with strong performance across all three categories.

![Classification Confusion Matrix](../figures/classification_confusion_matrix.png)

### Feature impact: academic calendar

Adding academic calendar flags (summer, breaks, finals) improved LightGBM regression RMSE from **0.145 to 0.128** on the test set — a meaningful gain that confirms campus schedule effects are real and worth modeling.

---

## 6. Conclusions and Recommendations

### Key findings

1. **Rec Center usage is highly predictable** from time, location, and calendar context. Our best model achieves ~9-point average error on a 0–1 scale for future time periods.
2. **Peak crowding is concentrated** on weekday afternoons (4–6 PM), especially Monday through Wednesday.
3. **Location matters significantly.** The Track Exercise Room and 2nd Floor run 60–80% higher utilization than the 1st Floor and Lower Exercise Room.
4. **Summer and breaks are reliably quieter**, making them good times for students who prefer less crowded workouts.
5. **Gradient boosting models are the practical choice** for deployment; neural networks and ensembles offer diminishing returns on this dataset.

### Recommendations for Rec Center staff and students

**For students:**
- Visit **early morning (before 10 AM)** or **weekends** for the lowest crowding.
- Avoid **Monday–Wednesday, 4:00–6:00 PM** if possible.
- If the Track Room or 2nd Floor is full, try the **Lower Exercise Room or 1st Floor** — they average roughly half the utilization.

**For Rec Center operations:**
- Use predicted peak windows to **align staffing and equipment availability** with expected demand.
- Publish a simple **low / medium / high forecast** by location and hour — the classification model supports this directly.
- Consider integrating predictions into the **Rec Center app or website** as a "busyness forecast" feature.

### Limitations

- Occuspace estimates are not exact headcounts; utilization above 100% suggests capacity list mismatches or sensor noise.
- The model reflects historical patterns and may not immediately capture sudden changes (new equipment, policy changes, or major campus events).
- Predictions are most reliable for **regular recurring patterns**; one-off events are not captured.

### Next steps

- Deploy a weekly-updated forecast dashboard by location and hour.
- Add real-time Occuspace API integration for live + predicted views.
- Re-train annually to incorporate new academic calendar years and facility changes.

---

*This report accompanies the full reproducible analysis in the [Rec Center GitHub repository](../README.md).*
