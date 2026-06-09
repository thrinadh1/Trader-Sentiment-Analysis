# 📈 Trader Sentiment Analysis: Fear vs Greed Impact on Crypto Trading Performance

## 🚀 Project Overview

Cryptocurrency markets are heavily influenced by investor psychology. Market sentiment indicators such as **Fear** and **Greed** often drive trading decisions, impacting profitability and risk exposure.

This project analyzes the relationship between **Bitcoin Market Sentiment (Fear & Greed Index)** and **Hyperliquid Trader Performance** to uncover patterns in trading behavior and profitability. The analysis combines sentiment data with historical trading records and applies statistical testing and machine learning techniques to generate actionable insights.

---

# 🎯 Objectives

### Business Objectives

* Understand how market sentiment affects trader profitability.
* Identify trading patterns during Fear and Greed market conditions.
* Discover profitable trading behaviors.
* Generate insights that can support trading strategies.

### Technical Objectives

* Clean and preprocess multiple datasets.
* Perform Exploratory Data Analysis (EDA).
* Conduct statistical hypothesis testing.
* Build machine learning models for trade outcome prediction.
* Generate reports and visualizations.

---

# 📊 Dataset Information

## Dataset 1: Bitcoin Market Sentiment

| Column         | Description                   |
| -------------- | ----------------------------- |
| timestamp      | Timestamp of sentiment record |
| value          | Fear & Greed score            |
| classification | Fear / Greed                  |
| date           | Observation date              |

---

## Dataset 2: Hyperliquid Historical Trader Data

| Column          | Description               |
| --------------- | ------------------------- |
| Account         | Trader Account ID         |
| Coin            | Cryptocurrency traded     |
| Execution Price | Trade execution price     |
| Size Tokens     | Quantity traded           |
| Size USD        | Trade value in USD        |
| Side            | Buy / Sell                |
| Timestamp       | Trade timestamp           |
| Start Position  | Position before execution |
| Direction       | Long / Short              |
| Closed PnL      | Realized Profit/Loss      |
| Transaction     | Transaction type          |
| Order ID        | Unique order identifier   |
| Crossed         | Order execution indicator |
| Fee             | Transaction fee           |
| Trade ID        | Unique trade identifier   |

---

# 🏗️ Project Architecture

```text
Trader_Sentiment_Analysis/
│
├── data/
│   ├── Bitcoin_Market_Sentiment.csv
│   └── Historical_Trader_Data.csv
│
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── coin_performance.csv
│   ├── top_traders.csv
│   └── final_report.txt
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🔄 Project Workflow

1. Data Loading
2. Data Cleaning
3. Data Integration
4. Feature Engineering
5. Exploratory Data Analysis
6. Statistical Analysis
7. Machine Learning
8. Feature Importance Analysis
9. Business Insights Generation

---

# ⚙️ Feature Engineering

The following features were created:

### Sentiment Encoding

| Sentiment | Encoding |
| --------- | -------- |
| Fear      | 0        |
| Greed     | 1        |

### Trade Outcome

* Win
* Loss

### Return Percentage

Return % = (Closed PnL / Size USD) × 100

### Target Variable

* 1 → Profitable Trade
* 0 → Loss-Making Trade

---

# 📈 Exploratory Data Analysis

The project includes:

### Market Sentiment Analysis

* Fear vs Greed Distribution
* Average Profitability by Sentiment
* Win Rate by Sentiment

### Trader Analysis

* Top 20 Most Profitable Traders
* Trader Performance Comparison

### Coin Analysis

* Coin-wise Profitability
* Coin Performance under Fear and Greed Conditions

### Trading Behavior Analysis

* Buy vs Sell Performance
* Direction-wise Analysis (Long vs Short)
* Fee Analysis

---

# 📊 Statistical Testing

## Independent T-Test

### Null Hypothesis (H₀)

Market sentiment has no significant effect on trader profitability.

### Alternative Hypothesis (H₁)

Market sentiment significantly affects trader profitability.

The project performs a T-Test between Fear and Greed market conditions using realized trade profits.

---

# 🤖 Machine Learning

## Problem Statement

Predict whether a trade will be profitable based on trade characteristics and market sentiment.

### Features Used

* Execution Price
* Size USD
* Fee
* Trade Side
* Market Sentiment

### Target

Trade Profitability

* 1 = Profit
* 0 = Loss

---

## Models Implemented

### Random Forest Classifier

Used as a baseline ensemble model.

### XGBoost Classifier

Used for advanced predictive modeling and feature importance analysis.

---

# 📏 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

---

# 📊 Visualizations Generated

* PnL Distribution by Sentiment
* Average PnL by Sentiment
* Feature Importance Chart
* Coin Performance Analysis
* Trader Profitability Analysis

---

# 📂 Outputs Generated

### Reports

* final_report.txt
* top_traders.csv
* coin_performance.csv
* side_analysis.csv
* direction_analysis.csv
* fee_analysis.csv

### Figures

* pnl_distribution.png
* avg_pnl_sentiment.png
* feature_importance.png

### Models

* trade_profit_model.pkl

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Statistical Analysis

* SciPy

### Machine Learning

* Scikit-Learn
* XGBoost

### Model Storage

* Joblib

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Trader-Sentiment-Analysis.git
```

Navigate to the project directory:

```bash
cd Trader-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python main.py
```

---

# 📋 Expected Results

After execution, the project will:

* Merge sentiment and trading datasets
* Generate analytical reports
* Produce visualizations
* Train machine learning models
* Identify profitable trading patterns
* Measure sentiment impact on performance

---

# 💡 Key Insights Delivered

* Impact of Fear vs Greed on trading profitability
* Trader behavior during different market conditions
* Most profitable coins and traders
* Trade characteristics associated with profitability
* Predictive power of market sentiment

---

# 🔮 Future Enhancements

* Interactive Streamlit Dashboard
* Power BI Dashboard
* Trader Segmentation using Clustering
* Sharpe Ratio Analysis
* Maximum Drawdown Analysis
* Hyperparameter Optimization
* Real-Time Sentiment Integration
* Live Trading Analytics

---

# 👨‍💻 Author

**Balathrinath Reddy**

Data Analyst | Machine Learning Enthusiast | Python | SQL | Power BI | Tableau

---

# 📜 License

This project is intended for educational, research, and portfolio purposes.
