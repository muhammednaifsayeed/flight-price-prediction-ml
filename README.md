# 🛫 Flight Price Prediction — End-to-End ML Regression Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Library-Pandas-green)
![R2 Score](https://img.shields.io/badge/R2%20Score-91.14%25-brightgreen)

## 📌 Executive Summary
This project implements an end-to-end Supervised Machine Learning Regression pipeline to predict Indian domestic flight ticket prices (`price`) using real-world flight features. The final tuned Ridge Regression model achieves an **R² score of 91.14%** with an average error (MAE) of **₹4,462**.

---

## 📊 Key Data Insights (EDA Findings)
- **Flight Class Impact:** `class` (Economy vs. Business) is the #1 price driver. Business class forms an upper price band (₹40,000–₹100,000+) with minimal overlap with Economy (under ₹20,000).
- **Target Skewness:** Ticket prices are heavily right-skewed (Mean ₹20,889 > Median ₹7,425) due to high-value Business Class tickets.
- **Surge Pricing Trend:** Prices remain relatively flat 50 to 4 days before departure, then spike non-linearly in the final 1–3 days.

---

## 🛠️ Machine Learning Pipeline Architecture
1. **Exploratory Data Analysis (EDA):** Histograms, Boxplots, Scatterplots, Lineplots, Pearson & Spearman correlation analysis.
2. **Feature Engineering:** Created `is_last_minute` binary indicator (`days_left <= 3`) based on domain insights.
3. **Preprocessing & Encoding:**
   - **Ordinal Encoding:** Applied to ordered categories (`class`, `stops`).
   - **One-Hot Encoding:** Applied to nominal text features (`airline`, `source_city`, `destination_city`, `departure_time`, `arrival_time`).
4. **Data Leakage Prevention:** Performed Train-Test Split (80/20) **before** applying `StandardScaler`.
5. **Model Training & Tuning:** Evaluated Linear, Ridge ($L_2$), and Lasso ($L_1$) regression; optimized using `GridSearchCV` with 5-Fold Cross-Validation.
6. **Model Serialization:** Exported trained model and scaler to `.pkl` files using `joblib`.

---

## 📈 Model Comparison Results

| Model | MAE (₹) | RMSE (₹) | R² Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | ₹4,462.83 | ₹6,758.30 | 91.14% |
| **Lasso Regression ($L_1$)** | ₹4,461.02 | ₹6,758.50 | 91.14% |
| **Tuned Ridge Regression ($L_2$)** | **₹4,462.81** | **₹6,758.30** | **91.14% (91.15% CV)** |

---

## 💻 How to Run Inference (Quick Start)

```python
import joblib
import pandas as pd

# Load saved assets
model = joblib.load('flight_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example input prediction
# (Pass encoded input DataFrame to model.predict())
