# PowerCo Customer Churn Prediction

A data science project investigating customer churn drivers for PowerCo, a major utility provider. Built predictive models to identify at-risk customers and challenged the hypothesis that price sensitivity is the primary churn driver.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📊 Project Overview

**Client:** PowerCo (Major Gas & Electricity Utility)  
**Industry:** Energy & Utilities  
**Problem:** Elevated customer churn in competitive SME market  
**Hypothesis:** Price sensitivity is driving customer attrition

### Key Results

- **Churn Rate:** 9.72% across 14,606 SME customers
- **Model Accuracy:** 83.85%
- **Model Recall:** 30.63%
- **Key Finding:** Price is NOT the primary churn driver
- **Top Predictors:** Net margin, consumption patterns, and customer tenure

## 🎯 Business Impact

**Challenge:** Client believed price sensitivity was causing customers to leave.

**Discovery:** Our analysis revealed that customer profitability (net margin) and consumption patterns are stronger predictors than price features.

**Opportunity:** Target high-value, at-risk customers for retention campaigns rather than engaging in price wars.

## 📁 Project Structure

```
powerco-churn-prediction/
│
├── 01_exploratory_data_analysis/
│   ├── powerco_eda.ipynb
│   └── README.md
│
├── 02_feature_engineering/
│   ├── powerco_feature_engineering.ipynb
│   └── README.md
│
├── 03_predictive_modeling/
│   ├── powerco_random_forest_model.ipynb
│   └── README.md
│
├── data/
│   ├── client_data.csv
│   ├── price_data.csv
│   └── data_for_predictions.csv
│
├── outputs/
│   ├── PowerCo_Executive_Summary.pptx
│   └── visualizations/
│
└── README.md
```

## 🔍 Analysis Workflow

### Task 1: Exploratory Data Analysis
- Analyzed 14,606 customer records with 44 features
- Identified 9.72% churn rate
- Discovered highly skewed distributions (consumption skew = 6.00)
- Found 25.5% missing sales channel data
- **Key Insight:** Contract type shows even split—may not be a churn driver

### Task 2: Feature Engineering
- Created 58 features from 44 original features
- Engineered price volatility metrics (avg, max, variance)
- Applied log transformations to reduce skewness
- Extracted temporal features (tenure, days to end)
- Removed 18 highly correlated features
- **Key Insight:** Price variance features created to capture customer frustration with bill fluctuations

### Task 3: Predictive Modeling
- Trained Random Forest classifier with GridSearchCV
- 5-fold cross-validation across 576 hyperparameter combinations
- Tested 2,880 models to find optimal configuration
- **Final Metrics:**
  - Accuracy: 83.85%
  - Precision: 24.03%
  - Recall: 30.63%
- **Key Insight:** Model excels at identifying non-churners but misses 70% of actual churners

## 📈 Key Findings

### 1. Price Sensitivity Hypothesis: REJECTED ❌

Price-related features did NOT emerge as top predictors. The hypothesis that price is the primary churn driver is not supported by the data.

### 2. Actual Churn Drivers (Feature Importance)

1. **Net Margin** - Customer profitability
2. **Consumption (12 months)** - Usage patterns
3. **Tenure** - Years as customer
4. **Forecasted Consumption** - Expected future usage

### 3. Model Limitations

- **Low Recall (30.63%):** Model misses ~70% of customers who will churn
- **Imbalanced Classes:** 90% non-churners vs 10% churners
- **Next Steps:** Model tuning needed to improve recall before deployment

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Visualization
- **scikit-learn** - Machine learning
  - RandomForestClassifier
  - GridSearchCV
  - train_test_split
  - StandardScaler
- **scipy** - Statistical analysis
- **Jupyter Notebook** - Interactive development

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/powerco-churn-prediction.git
cd powerco-churn-prediction

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# Launch Jupyter Notebook
jupyter notebook
```

### Running the Analysis

1. **Exploratory Data Analysis**
   ```bash
   cd 01_exploratory_data_analysis
   jupyter notebook powerco_eda.ipynb
   ```

2. **Feature Engineering**
   ```bash
   cd 02_feature_engineering
   jupyter notebook powerco_feature_engineering.ipynb
   ```

3. **Predictive Modeling**
   ```bash
   cd 03_predictive_modeling
   jupyter notebook powerco_random_forest_model.ipynb
   ```

## 📊 Results Summary

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 83.85% | Overall correct predictions |
| Precision | 24.03% | Of predicted churners, 24% actually churned |
| Recall | 30.63% | Of actual churners, 31% were identified |
| F1-Score | 26.92% | Harmonic mean of precision and recall |

### Business Translation

- **True Positives:** 87 customers correctly identified as churners
- **False Negatives:** 197 churners missed (opportunity cost)
- **False Positives:** 275 non-churners flagged (wasted retention effort)
- **True Negatives:** 2,363 correctly identified as staying

### ROI Potential

Assuming:
- Average customer lifetime value: £50,000
- Retention campaign cost: £500 per customer
- Retention success rate: 30%

**Current State (30% recall):**
- Identify: 426 at-risk customers annually
- Successful retentions: 128 customers
- Value saved: £6.4M
- Net benefit: ~£6.2M

**Target State (60% recall):**
- Net benefit: ~£12.4M
- **Incremental value: £6.2M annually**

## 💡 Key Recommendations

### Immediate Actions
1. **Do NOT compete on price alone** - Focus on customer value and service quality
2. **Target high-value, at-risk customers** - Prioritize retention resources
3. **Improve model recall** - Hyperparameter tuning and ensemble methods

### Strategic Initiatives
1. **Customer profitability management** - Focus on margin optimization
2. **Consumption trend monitoring** - Identify usage pattern changes early
3. **Tenure-based loyalty programs** - Strengthen customer relationships over time

### Technical Next Steps
1. Model improvement (target 50-60% recall)
2. Feature engineering with customer service data
3. Real-time churn risk dashboard
4. A/B test retention strategies

## 📝 Methodology

This project follows the BCG X data science methodology:

1. **Business Understanding & Problem Framing** - Defined churn drivers hypothesis
2. **Exploratory Data Analysis** - Analyzed distributions, correlations, and patterns
3. **Feature Engineering** - Created price volatility and temporal features
4. **Modeling & Evaluation** - Trained Random Forest with hyperparameter optimization
5. **Insights & Recommendations** - Translated findings into business actions

## 🤝 Contributing

This is a completed case study project. Feel free to fork and adapt for your own learning!

## 📧 Contact

**Project Author:** [Your Name]  
**LinkedIn:** [Your LinkedIn]  
**Email:** [Your Email]

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **BCG X** - For the project framework and case study structure
- **PowerCo** - Client case study (anonymized)
- **scikit-learn** - Machine learning library
- **Jupyter** - Interactive computing environment

---

**Note:** This project was completed as part of a BCG X data science virtual experience program. All data is synthetic and used for educational purposes.

⭐ If you found this project helpful, please star the repository!
