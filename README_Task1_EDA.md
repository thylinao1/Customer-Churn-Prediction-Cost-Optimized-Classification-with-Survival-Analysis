# Task 1: Exploratory Data Analysis

**Objective:** Investigate PowerCo customer data to understand churn patterns and test the hypothesis that price sensitivity drives customer attrition.

## 📊 Dataset Overview

### Customer Data (client_data.csv)
- **Rows:** 14,606 SME customers
- **Columns:** 44 features
- **Target Variable:** `churn` (1 = churned, 0 = active)

### Price Data (price_data.csv)
- Historical pricing at granular time intervals
- Variable and fixed pricing components
- Off-peak, peak, and mid-peak periods

## 🔍 Key Analyses Performed

### 1. Data Quality Assessment

```python
# Missing values identified
channel_sales: 3,725 missing (25.50%) ⚠️
origin_up: 64 missing (0.44%)
```

**Action:** Converted 'MISSING' strings to NaN for proper handling.

### 2. Target Variable Analysis

**Churn Rate: 9.72%**
- Churned customers: 1,419 (9.72%)
- Active customers: 13,187 (90.28%)
- **Imbalanced dataset** - Important for modeling strategy

**Insight:** While 9.72% is actually a good churn rate (lower is better), it presents a class imbalance challenge for predictive modeling.

### 3. Categorical Variable Analysis

#### Churn by Sales Channel
| Channel | Churn Rate | Customers | Churned |
|---------|------------|-----------|---------|
| foosdfp... | 12.1% | 6,754 | 820 |
| usilxup... | 10.0% | 1,375 | 138 |
| ewpakwl... | 8.4% | 893 | 75 |
| lmkebam... | 5.6% | 1,843 | 103 |
| Others | 0.0% | 16 | 0 |

**Key Finding:** Some sales channels have 12% churn while others have 0% churn. Sales channel IS an important factor.

#### Churn by Contract Type (Gas Service)
| Has Gas | Churn Rate | Interpretation |
|---------|------------|----------------|
| No Gas | 10.1% | Electricity only |
| Has Gas | 8.2% | Electricity + Gas |

**Key Finding:** Only 1.9 percentage point difference. Contract type may NOT be a major driving factor.

#### Churn by Number of Products
| Products | Churn Rate |
|----------|------------|
| 1 product | 10.0% |
| 2 products | 8.5% |
| 3 products | 9.9% |
| 4 products | 10.0% |

**Key Finding:** Churn rates are relatively consistent (8.5-10%). Number of products may NOT be a major churn driver.

#### Churn by Tenure (Years)
| Tenure | Churn Rate | Pattern |
|--------|------------|---------|
| 0-2 years | ~12% | Higher churn |
| 3-5 years | ~10% | Moderate |
| 6-8 years | ~8% | Lower |
| 9+ years | ~6% | Lowest |

**Key Finding:** Clear inverse relationship. Tenure appears to be a STRONG predictor of churn. Newer customers churn more.

### 4. Distribution Analysis

#### Consumption (cons_12m)
```
Mean:           159,220 kWh
Median:          14,116 kWh
Skewness:          6.00  (Highly positive skewed)
Outliers:       2,084 (14.27%)
```

**Visualization:** Distribution heavily skewed towards lower values. Most customers consume relatively little; few consume very large amounts.

**Action Required:** 
- Log transformation needed to normalize distribution
- Consider outlier treatment strategy

#### Subscribed Power (pow_max)
```
Mean:     47.97
Median:   43.65
Skewness:  5.79  (Highly positive skewed)
```

**Action Required:** Also needs transformation

#### Forecast Consumption (forecast_cons_year)
```
Mean:      6,429 kWh
Skewness:  16.59  (Extremely skewed)
```

**Action Required:** Most severely skewed feature

### 5. Price Sensitivity Analysis

**Correlation with Churn:**
```
forecast_price_energy_off_peak:  +0.144 (Weak)
forecast_price_energy_peak:      +0.011 (Very weak)
forecast_price_pow_off_peak:     -0.004 (Negligible)
```

**Key Finding:** All price features show WEAK correlations with churn. This does NOT strongly support the hypothesis that price is the primary driver.

⚠️ **Note:** This is static correlation. Need temporal price change analysis in feature engineering.

## 📈 Visualizations Created

1. **Churn Distribution Bar Chart**
   - 90% vs 10% split visualization
   - Highlights class imbalance

2. **Churn by Categorical Variables**
   - Sales channel comparison
   - Contract type comparison  
   - Product count comparison
   - Tenure analysis
   - Origin/offer comparison

3. **Distribution Plots**
   - Consumption distribution (positive skew visualization)
   - Subscribed power distribution
   - Forecast consumption distribution

4. **Boxplots for Outlier Detection**
   - Consumption outliers (14.27% detected)
   - Subscribed power outliers
   - Net margin outliers

## 💡 Key Insights Summary

### Strong Churn Indicators
✅ **Sales Channel** - Variation from 0% to 12% churn  
✅ **Customer Tenure** - Clear inverse relationship  
✅ **Origin/Contract Offer** - 6% to 12.6% variation  

### Weak Churn Indicators
❌ **Contract Type (Gas)** - Even split (10.1% vs 8.2%)  
❌ **Number of Products** - Consistent across counts  
❌ **Price Features** - All correlations < 0.15  

### Data Quality Issues
⚠️ **25.5% missing sales channel data** - Requires client consultation  
⚠️ **Highly skewed distributions** - Need transformations  
⚠️ **14.27% outliers** - Need outlier strategy  

## 🎯 Recommendations for Next Steps

### Feature Engineering (Task 2)
1. **Transform skewed features:**
   - Apply log(1+x) to consumption features
   - Normalize distributions for better model performance

2. **Handle outliers:**
   - Options: Remove, cap (winsorization), or use robust models
   - Document treatment strategy

3. **Create price change features:**
   - Calculate price differences over time
   - Compute price volatility metrics
   - Engineer "surprise bill" indicators

4. **Extract temporal features:**
   - Convert dates to month, season
   - Calculate tenure metrics
   - Days until contract end

5. **Encode categorical variables:**
   - One-hot encoding for sales channel, origin
   - Binary encoding for has_gas

### Modeling Strategy (Task 3)
1. **Handle class imbalance:**
   - Use stratified sampling
   - Consider SMOTE or class weighting
   - Choose appropriate metrics (not just accuracy!)

2. **Model selection:**
   - Random Forest (handles non-linearity, outliers)
   - XGBoost (typically best performance)
   - Ensemble methods

3. **Evaluation metrics:**
   - Focus on Recall (catching churners is critical)
   - Use AUC-ROC, F1-score
   - Analyze confusion matrix

## 📊 Statistical Summary

### Numeric Features
| Feature | Mean | Median | Std Dev | Skewness |
|---------|------|--------|---------|----------|
| cons_12m | 159,220 | 14,116 | 734,493 | 6.00 |
| pow_max | 47.97 | 43.65 | 59.60 | 5.79 |
| net_margin | 232.29 | 182.02 | 308.63 | 4.49 |
| num_years_antig | 5.04 | 5.00 | 2.95 | 0.30 |

### Categorical Features
| Feature | Unique Values | Mode | Mode Frequency |
|---------|---------------|------|----------------|
| channel_sales | 8 | foosdfp... | 46.2% |
| has_gas | 2 | f (No gas) | 81.8% |
| origin_up | 6 | lxidpid... | 48.6% |

## 🛠️ Code Highlights

### Loading and Initial Exploration
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
client_data = pd.read_csv('client_data.csv')
price_data = pd.read_csv('price_data.csv')

# Basic info
print(f"Shape: {client_data.shape}")
print(f"Churn rate: {client_data['churn'].mean()*100:.2f}%")
```

### Handling Missing Values
```python
# Convert MISSING strings to NaN
client_data['channel_sales'] = client_data['channel_sales'].replace('MISSING', np.nan)
client_data['origin_up'] = client_data['origin_up'].replace('MISSING', np.nan)
```

### Analyzing Churn by Categories
```python
# Churn rate by sales channel
channel_churn = client_data.groupby('channel_sales')['churn'].agg(['sum', 'count', 'mean'])
channel_churn.sort_values('mean', ascending=False)
```

### Detecting Skewness
```python
# Calculate skewness
from scipy import stats
skewness = client_data['cons_12m'].skew()
print(f"Skewness: {skewness:.2f}")
# Output: 6.00 (highly positive skewed)
```

### Outlier Detection
```python
# IQR method
Q1 = client_data['cons_12m'].quantile(0.25)
Q3 = client_data['cons_12m'].quantile(0.75)
IQR = Q3 - Q1
outliers = client_data[(client_data['cons_12m'] < Q1 - 1.5*IQR) | 
                       (client_data['cons_12m'] > Q3 + 1.5*IQR)]
print(f"Outliers: {len(outliers)} ({len(outliers)/len(client_data)*100:.2f}%)")
```

## 📁 Files Generated

- `powerco_eda.ipynb` - Complete exploratory analysis notebook
- `eda_visualizations.png` - Key charts and graphs
- `data_quality_report.txt` - Missing values and outliers summary

## ⏭️ Next Steps

Proceed to **Task 2: Feature Engineering** to:
1. Transform skewed distributions
2. Create price volatility features
3. Extract temporal patterns
4. Encode categorical variables
5. Prepare dataset for modeling

---

**Conclusion:** EDA revealed that price features show weak correlations with churn, challenging the initial hypothesis. Tenure, sales channel, and customer profitability appear to be stronger indicators.
