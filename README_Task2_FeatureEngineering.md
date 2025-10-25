# Task 2: Feature Engineering

**Objective:** Enrich the dataset through feature engineering to improve churn prediction accuracy. Following Estelle's suggestion: "The difference between off-peak prices in December and January could be a significant feature."

## 🎯 Feature Engineering Strategy

Following best practices framework:
1. **Remove** unnecessary columns
2. **Expand** datasets with existing columns
3. **Combine** columns to create "better" features
4. **Transform** distributions for model assumptions
5. **Encode** categorical variables

## 📊 Transformation Summary

### Before vs After
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total Features | 44 | 58 | +14 net |
| Features Created | - | 46 | New |
| Features Removed | - | 32 | Redundant |
| Missing Values | Yes | No | Handled |

## 🔧 Feature Engineering Steps

### Step 1: Column Removal

**Removed:** `id` column
- **Reason:** Unique identifier with no predictive value
- **Impact:** Can cause overfitting if included

### Step 2: Date Feature Extraction

#### Raw Date Columns (Not Useful for Models)
- `date_activ` - Activation date
- `date_end` - Contract end date
- `date_modif_prod` - Last product modification
- `date_renewal` - Renewal date

#### Extracted Features (Actionable)

**Seasonal Features:**
```python
df['month_activ'] = df['date_activ'].dt.month
df['month_end'] = df['date_end'].dt.month
df['month_modif_prod'] = df['date_modif_prod'].dt.month
df['month_renewal'] = df['date_renewal'].dt.month
```
- Captures seasonality in signup/churn patterns
- Values: 1-12 (January to December)

**Temporal Metrics:**
```python
reference_date = pd.to_datetime('2016-01-01')
df['tenure_days'] = (reference_date - df['date_activ']).dt.days
df['days_to_end'] = (df['date_end'] - reference_date).dt.days
df['days_since_modif'] = (reference_date - df['date_modif_prod']).dt.days
```

| Feature | Mean | Purpose |
|---------|------|---------|
| tenure_days | 1,799 days (~5 years) | Customer loyalty indicator |
| days_to_end | 209 days | Contract urgency metric |
| days_since_modif | 1,093 days (~3 years) | Engagement indicator |

**Rationale:** Raw dates mean nothing to models, but extracted metrics capture:
- Customer lifecycle stage
- Seasonality patterns
- Engagement levels

### Step 3: Price Feature Engineering ⭐

Following Estelle's insight about price differences, created features to capture price volatility:

#### Average Price Changes
```python
# Overall price trend
df['avg_price_change_year'] = df[price_year_cols].mean(axis=1)
df['avg_price_change_6m'] = df[price_6m_cols].mean(axis=1)
```
- Mean: 1.23 (yearly), 1.01 (6-month)
- Captures overall price movement

#### Maximum Price Changes (Volatility/Shock)
```python
# Price spikes that frustrate customers
df['max_price_change_year'] = df[price_year_cols].max(axis=1)
df['max_price_change_6m'] = df[price_6m_cols].max(axis=1)
```
- Mean: 2.55 (yearly), 2.05 (6-month)
- **Rationale:** If your utility bill suddenly spikes in winter, you're annoyed and want a better deal!

#### Minimum Price Changes
```python
df['min_price_change_year'] = df[price_year_cols].min(axis=1)
df['min_price_change_6m'] = df[price_6m_cols].min(axis=1)
```
- Captures discounting or stable periods

#### Price Variance (Stability Metric)
```python
df['price_variance_year'] = df[price_year_cols].std(axis=1)
df['price_variance_6m'] = df[price_6m_cols].std(axis=1)
```
- Mean: 1.24 (yearly), 0.98 (6-month)
- **Rationale:** Volatile prices = frustrated customers

**Business Logic:** Average change alone doesn't tell the full story. Customers may tolerate gradual increases but churn when prices are unpredictable.

### Step 4: Consumption & Margin Features

#### Consumption Trend
```python
# Is usage increasing or decreasing?
df['cons_change_rate'] = (df['cons_last_month'] * 12) / (df['cons_12m'] + 1)
```
- Ratio > 1: Usage increasing
- Ratio < 1: Usage decreasing
- Indicates lifestyle changes (moved, business growth/decline)

#### Forecast Accuracy
```python
# How good are our predictions?
df['forecast_vs_actual_cons'] = df['forecast_cons_12m'] / (df['cons_12m'] + 1)
```
- Ratio > 1: We over-predicted
- Ratio < 1: We under-predicted
- Billing surprises may trigger churn

#### Profitability Metrics
```python
# Revenue per unit consumed
df['margin_per_kwh'] = df['net_margin'] / (df['cons_12m'] + 1)

# Gap between gross and net
df['gross_net_margin_diff'] = df['margin_gross_pow_ele'] - df['margin_net_pow_ele']
```
- Identifies low-profit customers (may be on bad deals)
- Higher churn risk if we're not making money on them

### Step 5: Categorical Encoding

#### Boolean to Binary
```python
# has_gas: 't' → 1, 'f' → 0
df['has_gas'] = df['has_gas'].map({'t': 1, 'f': 0})
```
- ML algorithms need numeric inputs

#### One-Hot Encoding
```python
# Create dummy variables
df = pd.get_dummies(df, columns=['channel_sales', 'origin_up'], 
                    drop_first=True, dummy_na=True)
```

**Before:**
- channel_sales: 8 categories
- origin_up: 6 categories

**After:**
- channel_sales_*: 7 binary columns
- origin_up_*: 5 binary columns
- _nan columns: Handle missing values

**Parameters:**
- `drop_first=True` - Avoids multicollinearity (dummy variable trap)
- `dummy_na=True` - Creates columns for missing values

### Step 6: Distribution Transformations ⭐

From EDA, we know consumption features are highly skewed. Many ML algorithms assume normally distributed features.

#### Log Transformations

| Feature | Skew Before | Skew After | Improvement |
|---------|-------------|------------|-------------|
| cons_12m | 6.00 | -0.38 | 5.62 ✓✓✓ |
| cons_gas_12m | 9.60 | 1.88 | 7.72 ✓✓✓ |
| cons_last_month | 6.39 | -0.19 | 6.20 ✓✓✓ |
| pow_max | 5.79 | 1.80 | 3.98 ✓✓ |
| forecast_cons_12m | 7.16 | -2.03 | 5.12 ✓✓✓ |
| forecast_cons_year | 16.59 | -0.12 | 16.47 ✓✓✓ |

```python
# Apply log(1 + x) transformation
for col in skewed_cols:
    df[f'{col}_log'] = np.log1p(df[col])
```

**Why log1p?**
- `log1p(x)` = log(1 + x)
- Handles zeros (log(0) is undefined, log(1) = 0)
- Compresses large values, expands small values

**Visual Impact:**
- Before: Extreme right tail, most data clustered at left
- After: Nearly bell-shaped distribution

**Benefits:**
- Better model performance (especially linear models)
- More interpretable relationships
- Reduced impact of extreme values

### Step 7: Correlation Analysis & Removal

#### High Correlations Detected (r > 0.95)

Found 30 pairs of highly correlated features:

**Examples:**
- `margin_gross_pow_ele` ↔ `margin_net_pow_ele`: r = 1.000 (perfect)
- `cons_last_month` ↔ `cons_12m`: r = 0.968
- `var_year_price_off_peak` ↔ `var_year_price_off_peak_fix`: r = 1.000

#### Removal Strategy
```python
# For each correlated pair, keep first feature, drop second
features_to_drop = set()
for feat1, feat2, corr in high_corr_pairs:
    if feat2 not in features_to_drop and feat2 != 'churn':
        features_to_drop.add(feat2)
```

**Removed:** 18 redundant features
- margin_gross_pow_ele (kept margin_net_pow_ele)
- cons_12m (kept cons_12m_log and cons_last_month)
- Multiple redundant price features

**Rationale:**
- Redundant information
- Causes multicollinearity (inflated standard errors)
- Risk of overfitting
- Wasted computational resources

## 📈 Feature Importance Preview

While full feature importance comes from modeling, we can anticipate based on correlation analysis:

**Expected High Importance:**
1. Net margin features
2. Consumption patterns
3. Tenure metrics
4. Price volatility indicators

**Expected Medium Importance:**
5. Seasonal features
6. Days to contract end
7. Forecast accuracy

**Expected Low Importance:**
8. Individual price point features (without volatility)

## 🎨 Visualization: Before vs After Transformations

### Consumption Distribution (cons_12m)

**Before Transformation:**
- Distribution: Extreme right skew
- Skewness: 6.00
- Mean >> Median (159,220 vs 14,116)
- Issues: Outliers dominate, poor for linear models

**After Log Transformation:**
- Distribution: Nearly normal
- Skewness: -0.38
- Mean ≈ Median
- Benefits: Better for modeling

## 💡 Key Engineering Decisions

### 1. Why Create Price Variance Features?
**Problem:** Customer says "my bill went up!"  
**Reality:** It's not the absolute increase, it's the unpredictability  
**Solution:** Capture volatility, not just average change

### 2. Why Keep Original AND Transformed Features?
**Flexibility:** Some models prefer original scale (tree-based), others prefer normalized (linear)  
**Decision:** Let feature selection or model choose

### 3. Why Drop Correlated Features?
**Example:** If `cons_12m` and `cons_last_month` are 96.8% correlated, they provide redundant information  
**Impact:** One is sufficient, keeping both adds noise

### 4. Why Log Transform?
**Math:** Multiplicative relationships become additive  
**Interpretation:** 10% increase in consumption → constant effect on churn regardless of baseline  
**Models:** Linear models, SVMs benefit most

## 📊 Final Dataset Characteristics

### Numeric Features: 52
- Original: 30
- Log transformed: 6
- Engineered: 16

### Categorical Features (After Encoding): 6
- Binary: 1 (has_gas)
- One-hot encoded: 12 dummy variables

### Target Variable: 1
- churn (0/1)

**Total:** 58 features ready for modeling

## 🛠️ Code Highlights

### Creating Price Volatility Features
```python
# Define price columns
price_year_cols = ['var_year_price_off_peak', 'var_year_price_peak', 
                   'var_year_price_mid_peak']

# Maximum price change (volatility indicator)
df['max_price_change_year'] = df[price_year_cols].max(axis=1)

# Price variance (stability indicator)
df['price_variance_year'] = df[price_year_cols].std(axis=1)
```

### Log Transformation with Skewness Check
```python
from scipy import stats

for col in skewed_cols:
    skew_before = df[col].skew()
    df[f'{col}_log'] = np.log1p(df[col])
    skew_after = df[f'{col}_log'].skew()
    improvement = abs(skew_before) - abs(skew_after)
    print(f"{col}: {skew_before:.2f} → {skew_after:.2f} (Δ {improvement:.2f})")
```

### Correlation-Based Feature Removal
```python
# Calculate correlation matrix
corr_matrix = df.corr().abs()

# Find upper triangle pairs > 0.95
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identify features to drop
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
df = df.drop(columns=to_drop)
```

## 📁 Files Generated

- `powerco_feature_engineering.ipynb` - Complete engineering notebook
- `data_for_modeling.csv` - Final feature-engineered dataset (14,606 × 58)
- `feature_engineering_transformations.png` - Before/after visualizations
- `feature_list.txt` - All 58 features with descriptions

## ✅ Quality Checks

### Data Integrity
- ✓ No missing values after encoding (dummy_na handles NaNs)
- ✓ All features are numeric
- ✓ No infinite values from transformations
- ✓ All 14,606 customers retained (no data loss)

### Feature Quality
- ✓ Distributions are more normal (skew reduced from 6-16 to <2)
- ✓ Redundancy removed (18 highly correlated features dropped)
- ✓ Business logic validated for all engineered features
- ✓ Proper encoding without multicollinearity

### Model Readiness
- ✓ Target variable (churn) balanced appropriately for stratified sampling
- ✓ Features scaled appropriately (log transform)
- ✓ Categorical variables encoded
- ✓ No leakage (all features available at prediction time)

## 🎯 Impact on Modeling

This feature engineering should improve model performance by:

1. **Capturing price sensitivity** - Volatility features reflect customer frustration
2. **Normalizing distributions** - Log transforms help linear models
3. **Reducing noise** - Removed 18 redundant features
4. **Creating interpretable features** - Tenure, margin per kWh, price variance
5. **Handling missing data** - Dummy_na preserves all records

## ⏭️ Next Steps

Proceed to **Task 3: Predictive Modeling** to:
1. Train Random Forest classifier
2. Perform hyperparameter tuning
3. Evaluate model performance
4. Analyze feature importance
5. Test price sensitivity hypothesis

---

**Conclusion:** Created 46 new features and removed 32 redundant ones, resulting in 58 high-quality features. Dataset is now optimized for machine learning with normalized distributions and engineered price volatility metrics.
