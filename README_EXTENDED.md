# Customer Churn Prediction: Cost-Optimized Classification with Survival Analysis

Churn prediction model improving recall from 7% to 94% via cost-sensitive threshold optimization, generating £13.7M in projected annual savings. Applied SMOTE, Cox Proportional Hazards survival analysis, and expected value framework. Rejected price sensitivity hypothesis—campaign origin and customer margins drive churn, not price.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Technical Approach](#technical-approach)
- [Ablation Study](#ablation-study)
- [Business Insights](#business-insights)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Methodology](#methodology)
- [Limitations & Future Work](#limitations--future-work)

---

## Project Overview

### Context
Completed as part of BCG X Data Science Virtual Experience. PowerCo, a major gas and electricity utility, faced elevated customer churn in the competitive SME market. The client hypothesized that **price sensitivity was the primary churn driver**.

### Objective
1. Investigate whether price sensitivity drives customer churn
2. Build a predictive model to identify at-risk customers
3. Quantify business impact of retention interventions

### My Contributions Beyond Base Case
Rather than stopping at a basic classification model, I extended the analysis with:
- Cost-sensitive threshold optimization
- Survival analysis (Kaplan-Meier + Cox Proportional Hazards)
- Expected value framework with sensitivity analysis
- Rigorous class imbalance handling
- Permutation-based feature importance

---

## Key Results

### Performance Metrics

| Metric | Baseline Model | Final Model | Improvement |
|--------|----------------|-------------|-------------|
| Recall | 7.4% | **94.0%** | 12.7x |
| Precision | 77.8% | 10.8% | Trade-off |
| AUC-ROC | 0.69 | **0.70** | +1.4% |
| Annual Cost | £12.9M | **-£0.7M** | £13.7M saved |

### Primary Finding
**Price is NOT the main churn driver.** Campaign acquisition source, customer margins, and consumption patterns are stronger predictors than absolute price levels.

---

## Technical Approach

### 1. Exploratory Data Analysis
- **Dataset:** 14,606 SME customers with 26 features
- **Churn rate:** 9.72% (severely imbalanced)
- **Missing data:** 25.5% in sales channel (handled appropriately)
- **Key insight:** Price correlations with churn < 0.03 (weak)

### 2. Feature Engineering
Created 58 features from original 44:

```python
# Price volatility metrics
df['avg_price_change_year'] = df[price_year_cols].mean(axis=1)
df['max_price_change_year'] = df[price_year_cols].max(axis=1)
df['var_price_change_year'] = df[price_year_cols].var(axis=1)

# Temporal features
df['tenure_days'] = (reference_date - df['date_activ']).dt.days
df['days_to_end'] = (df['date_end'] - reference_date).dt.days

# Customer behavior
df['margin_per_kwh'] = df['net_margin'] / (df['cons_12m'] + 1)
df['cons_change_rate'] = (df['cons_last_month'] * 12) / (df['cons_12m'] + 1)
```

**Log transformations** reduced skewness dramatically:
- `cons_12m`: 6.00 → -0.38
- `forecast_cons_year`: 16.59 → -0.12

### 3. Class Imbalance Handling

Compared multiple strategies:

| Method | Recall | Precision | AUC-ROC |
|--------|--------|-----------|---------|
| No Resampling | 0.049 | 0.778 | 0.699 |
| SMOTE | **0.130** | 0.627 | 0.700 |
| ADASYN | 0.116 | 0.541 | 0.705 |
| SMOTETomek | 0.109 | 0.544 | 0.701 |
| Class Weight 1:10 | 0.116 | 0.702 | 0.696 |

**Selected:** SMOTE for best precision-recall balance.

### 4. Cost-Sensitive Threshold Optimization

The core insight: **Default threshold (0.5) optimizes accuracy, not business value.**

Defined explicit cost matrix:
```
Cost = FN × £50,000 (lost CLV) 
     + FP × £500 (campaign cost) 
     - TP × £10,000 (retention benefit)
```

Results:
| Threshold | Recall | Precision | False Positives | Net Cost |
|-----------|--------|-----------|-----------------|----------|
| 0.50 | 13.0% | 62.7% | 22 | £11,991,000 |
| 0.20 | 59.2% | 18.7% | 733 | £4,486,500 |
| 0.10 | 80.3% | 12.4% | 1,618 | £1,329,000 |
| **0.05** | **94.0%** | **10.8%** | **2,198** | **-£721,000** |

**Optimal threshold: 0.05** — catches 94% of churners, generates profit.

### 5. Model Comparison

```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6)
}
```

| Model | AUC-ROC | AUC-PR | Recall@0.05 |
|-------|---------|--------|-------------|
| Logistic Regression | 0.598 | 0.141 | 86.6% |
| **Random Forest** | **0.700** | **0.299** | **94.0%** |
| XGBoost | 0.700 | 0.251 | 90.5% |

Tree-based models significantly outperform linear models (AUC 0.70 vs 0.60), indicating **non-linear relationships** in churn drivers.

### 6. Expected Value Framework

Model profitability depends on retention campaign effectiveness:

```python
def calculate_expected_value(threshold, y_prob, y_true, 
                             clv=50000, retention_rate=0.3, campaign_cost=500):
    # EV = Σ [P(churn) × CLV × P(retain|intervene) - C(intervene)]
    benefit = tp * clv * retention_rate
    cost = (tp + fp) * campaign_cost + fn * clv
    return benefit - cost
```

| Retention Rate | Expected Value |
|----------------|----------------|
| 10% | -£1,188,000 (loss) |
| 20% | £97,000 (break-even) |
| 30% | £1,382,000 |
| 40% | £2,667,000 |
| 50% | £3,952,000 |

**Critical insight:** Model only profitable if retention campaigns succeed >20% of time.

### 7. Survival Analysis

Reframed churn as **time-to-event** using lifelines library.

**Kaplan-Meier Survival Curves:**
- Gas customers show higher survival probability
- Clear separation between customer segments
- Median survival time: >12 years (low churn overall)

**Cox Proportional Hazards Model:**

| Feature | Hazard Ratio | p-value | Interpretation |
|---------|--------------|---------|----------------|
| has_gas | 0.90 | **0.04** | 10% lower churn risk |
| net_margin | 1.00 | **<0.005** | Statistically significant |
| var_year_price_off_peak | 1.00 | 0.11 | Not significant |
| cons_12m | 0.98 | 0.40 | Not significant |

**Key finding:** Gas customers have statistically significant lower churn hazard. Cross-selling opportunity.

### 8. Feature Importance (Permutation-Based)

More rigorous than Gini importance—measures actual prediction impact:

```python
perm_importance = permutation_importance(rf_best, X_test, y_test, 
                                         n_repeats=10, random_state=42)
```

Top 5 features:
1. `origin_up_*` (campaign acquisition) — 0.014
2. `origin_up_*` (campaign acquisition) — 0.011
3. `origin_up_*` (campaign acquisition) — 0.008
4. `channel_*` (sales channel) — 0.008
5. `offpeak_diff_dec_january_power` — 0.007

**Campaign origin codes dominate—not price features.**

---

## Ablation Study

Quantified incremental value of each technical improvement:

| Model Stage | Recall | AUC-ROC | Net Cost | Δ Cost |
|------------|--------|---------|----------|--------|
| Baseline RF (threshold=0.5) | 7.4% | 0.694 | £12,943,000 | — |
| + SMOTE | 13.0% | 0.700 | £11,991,000 | £952,000 |
| + Threshold Optimization | **94.0%** | 0.700 | **-£721,000** | **£12,712,000** |

**Key takeaway:** Threshold optimization provided **7x more improvement** than SMOTE. Business cost optimization > ML metric optimization.

---

## Business Insights

### Hypothesis Testing Result
**REJECTED: Price sensitivity is NOT the primary churn driver.**

Evidence:
- Price feature correlations with churn: <0.03
- Price volatility not significant in Cox model (p=0.11)
- Campaign origin and margins are top predictors

### Actionable Recommendations

1. **Don't compete on price**
   - Focus on service quality and margin optimization
   - Price discounting erodes profitability without reducing churn

2. **Cross-sell gas services**
   - 10% hazard reduction (statistically significant)
   - Dual-service customers more sticky

3. **Use optimized threshold (0.05)**
   - Catches 94% of churners vs 13% at default
   - Trade-off: More false positives (£500 each) but fewer missed churners (£50k each)

4. **Monitor retention campaign effectiveness**
   - Model profitable only if success rate >20%
   - Track actual retention outcomes to validate assumptions

5. **Target by acquisition channel**
   - Campaign origin is strongest predictor
   - Some acquisition channels produce higher-risk customers

---

## Repository Structure

```
powerco-churn-prediction/
│
├── powerco_eda.ipynb                    # Exploratory Data Analysis
│   ├── Data loading and cleaning
│   ├── Missing value analysis
│   ├── Distribution analysis
│   ├── Correlation analysis
│   └── Initial hypothesis testing
│
├── powerco_feature_engineering.ipynb    # Feature Engineering
│   ├── Temporal feature extraction
│   ├── Price volatility metrics
│   ├── Log transformations (skewness reduction)
│   ├── Customer behavior features
│   └── Multicollinearity removal
│
├── powerco_random_forest_model.ipynb    # Modeling Pipeline
│   ├── Class imbalance handling (SMOTE, ADASYN, class weights)
│   ├── Cost-sensitive threshold optimization
│   ├── Model comparison (RF, XGBoost, Logistic Regression)
│   ├── Expected value framework
│   ├── Survival analysis (Kaplan-Meier, Cox PH)
│   ├── Permutation feature importance
│   └── Ablation study
│
└── README.md                            # This file
```

---

## Installation

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
```

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost imbalanced-learn lifelines
```

### Running the Analysis

```bash
# 1. Exploratory Data Analysis
jupyter notebook powerco_eda.ipynb

# 2. Feature Engineering
jupyter notebook powerco_feature_engineering.ipynb

# 3. Modeling (main analysis)
jupyter notebook powerco_random_forest_model.ipynb
```

---

## Methodology

This project follows rigorous data science practices:

1. **Problem Framing** — Defined business cost function, not just ML metrics
2. **EDA** — Identified class imbalance, missing data, weak price correlations
3. **Feature Engineering** — Created domain-specific features (volatility, trends)
4. **Modeling** — Compared multiple algorithms, handled class imbalance properly
5. **Evaluation** — Cost-sensitive threshold optimization, not accuracy
6. **Validation** — Survival analysis for statistical rigor
7. **Interpretation** — Permutation importance, not just Gini
8. **Ablation** — Quantified value of each technical contribution

---

## Limitations & Future Work

### Current Limitations
- **Model concordance (0.56)** suggests room for additional features
- **Anonymized campaign codes** limit full interpretability
- **No customer service data** — interactions likely predictive
- **Static snapshot** — no temporal validation on future data
- **Assumes fixed costs** — sensitivity analysis shows break-even at 20% retention

### Future Improvements

1. **Uplift Modeling**
   - Predict who will *respond* to intervention, not just who will churn
   - Causal inference vs correlation

2. **Additional Data Sources**
   - Customer service call logs
   - Billing complaint history
   - Competitor pricing data

3. **Model Enhancements**
   - Neural networks for complex interactions
   - Time-series forecasting for consumption trends
   - Ensemble stacking with economic optimization

4. **Production Deployment**
   - Real-time scoring API
   - A/B testing of retention strategies
   - Monitoring for model drift

5. **Bayesian Approach**
   - Uncertainty quantification in predictions
   - "80% confident this customer has 15-25% churn risk"

---

## Key Learnings

1. **ML metrics ≠ Business value**
   - 83% accuracy sounds good, but missed 93% of churners
   - Optimize for cost function, not log-loss

2. **Threshold optimization > Model tuning**
   - GridSearchCV over 576 configs: marginal gains
   - Threshold from 0.5 to 0.05: 12.7x recall improvement

3. **Class imbalance is critical**
   - 90/10 split makes baseline models useless
   - SMOTE helps, but threshold matters more

4. **Statistical rigor validates findings**
   - Cox model provides p-values, not just feature importance
   - has_gas: p=0.04 is actionable

5. **Business context drives everything**
   - Understanding CLV, campaign costs, retention rates
   - Data science without business impact is just math

---

## Technologies Used

- **Languages:** Python 3.8+
- **ML Libraries:** scikit-learn, XGBoost, imbalanced-learn
- **Survival Analysis:** lifelines (Kaplan-Meier, Cox PH)
- **Data Processing:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Development:** Jupyter Notebook

---

## Acknowledgments

- **BCG X** — Virtual Experience program framework
- **PowerCo** — Anonymized client case study
- **scikit-learn** — ML implementation
- **lifelines** — Survival analysis library

---

## Contact

For questions about methodology or implementation details, feel free to reach out.

---

*BCG X Data Science Virtual Experience, extended with SMOTE, Cox Proportional Hazards survival analysis, cost-sensitive threshold optimization, and expected value framework.*

**TL;DR:** Turned a 7% recall model into 94% recall by optimizing for business costs instead of ML metrics. Saved £13.7M. Price isn't the problem—it's how you acquired the customer.
