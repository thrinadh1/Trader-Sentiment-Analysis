import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==========================================================
# CONFIGURATION
# ==========================================================

SENTIMENT_FILE = "data/fear_greed_index.csv"
TRADER_FILE = "data/historical_data.csv"

OUTPUT_DIR = "outputs"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

sentiment_df = pd.read_csv(SENTIMENT_FILE)
trader_df = pd.read_csv(TRADER_FILE)

print("Sentiment Shape :", sentiment_df.shape)
print("Trader Shape    :", trader_df.shape)

# ==========================================================
# DATA CLEANING
# ==========================================================

print("\nCleaning Data...")

sentiment_df.columns = sentiment_df.columns.str.strip()
trader_df.columns = trader_df.columns.str.strip()

sentiment_df.drop_duplicates(inplace=True)
trader_df.drop_duplicates(inplace=True)

# Sentiment Dataset

sentiment_df["date"] = pd.to_datetime(
    sentiment_df["date"],
    errors="coerce"
)

sentiment_df["date"] = sentiment_df["date"].dt.date

# Trader Dataset

trader_df["Timestamp"] = pd.to_datetime(
    trader_df["Timestamp"],
    errors="coerce"
)

trader_df["Date"] = trader_df["Timestamp"].dt.date

# ==========================================================
# MERGE DATASETS
# ==========================================================

print("\nMerging Data...")

df = trader_df.merge(
    sentiment_df,
    left_on="Date",
    right_on="date",
    how="left"
)

print("Merged Shape :", df.shape)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

print("\nCreating Features...")

df["classification"] = (
    df["classification"]
    .fillna("Unknown")
)

sentiment_map = {
    "Fear": 0,
    "Greed": 1
}

df["Sentiment_Encoded"] = (
    df["classification"]
    .map(sentiment_map)
)

df["Trade_Result"] = np.where(
    df["Closed PnL"] > 0,
    "Win",
    "Loss"
)

trade_value = (
    df["Size USD"]
    .replace(0, 1)
)

df["Return_Pct"] = (
    df["Closed PnL"] /
    trade_value
) * 100

df["Target"] = np.where(
    df["Closed PnL"] > 0,
    1,
    0
)

# ==========================================================
# BASIC DATA SUMMARY
# ==========================================================

print("\nDataset Summary")

print(df.info())

print(df.describe())

# ==========================================================
# WIN RATE
# ==========================================================

print("\nWin Rate By Sentiment")

win_rate = (
    df.groupby("classification")
    ["Trade_Result"]
    .apply(
        lambda x:
        (x == "Win").mean() * 100
    )
)

print(win_rate)

win_rate.to_csv(
    f"{OUTPUT_DIR}/win_rate.csv"
)

# ==========================================================
# PNL BY SENTIMENT
# ==========================================================

avg_pnl = (
    df.groupby("classification")
    ["Closed PnL"]
    .mean()
)

print("\nAverage PnL")

print(avg_pnl)

avg_pnl.to_csv(
    f"{OUTPUT_DIR}/avg_pnl_sentiment.csv"
)

# ==========================================================
# VISUALIZATION 1
# ==========================================================

plt.figure(figsize=(10,6))

sns.boxplot(
    x="classification",
    y="Closed PnL",
    data=df
)

plt.title(
    "PnL Distribution By Market Sentiment"
)

plt.savefig(
    f"{FIGURE_DIR}/pnl_distribution.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# VISUALIZATION 2
# ==========================================================

plt.figure(figsize=(10,6))

avg_pnl.plot(
    kind="bar"
)

plt.title(
    "Average PnL By Sentiment"
)

plt.ylabel("Average PnL")

plt.savefig(
    f"{FIGURE_DIR}/avg_pnl_sentiment.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# TOP TRADERS
# ==========================================================

print("\nTop Traders")

top_traders = (
    df.groupby("Account")
    ["Closed PnL"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(20)
)

print(top_traders)

top_traders.to_csv(
    f"{OUTPUT_DIR}/top_traders.csv"
)

# ==========================================================
# COIN PERFORMANCE
# ==========================================================

coin_perf = (
    df.groupby(
        ["Coin", "classification"]
    )["Closed PnL"]
    .mean()
    .reset_index()
)

coin_perf.to_csv(
    f"{OUTPUT_DIR}/coin_performance.csv",
    index=False
)

# ==========================================================
# BUY / SELL ANALYSIS
# ==========================================================

side_analysis = (
    df.groupby("Side")
    ["Closed PnL"]
    .mean()
)

print("\nSide Analysis")

print(side_analysis)

side_analysis.to_csv(
    f"{OUTPUT_DIR}/side_analysis.csv"
)

# ==========================================================
# DIRECTION ANALYSIS
# ==========================================================

direction_analysis = (
    df.groupby("Direction")
    ["Closed PnL"]
    .mean()
)

direction_analysis.to_csv(
    f"{OUTPUT_DIR}/direction_analysis.csv"
)

# ==========================================================
# FEES ANALYSIS
# ==========================================================

fee_analysis = (
    df.groupby("classification")
    ["Fee"]
    .mean()
)

fee_analysis.to_csv(
    f"{OUTPUT_DIR}/fee_analysis.csv"
)

# ==========================================================
# T-TEST
# ==========================================================

print("\nT-Test")

fear_pnl = df[
    df["classification"] == "Fear"
]["Closed PnL"]

greed_pnl = df[
    df["classification"] == "Greed"
]["Closed PnL"]

if len(fear_pnl) > 5 and len(greed_pnl) > 5:

    stat, p = ttest_ind(
        fear_pnl,
        greed_pnl,
        nan_policy="omit"
    )

    print("T Statistic :", stat)
    print("P Value     :", p)

# ==========================================================
# MACHINE LEARNING
# ==========================================================

print("\nMachine Learning")

encoder = LabelEncoder()

df["side_encoded"] = (
    encoder.fit_transform(
        df["Side"].astype(str)
    )
)

features = [
    "Execution Price",
    "Size USD",
    "Fee",
    "side_encoded",
    "Sentiment_Encoded"
]

X = df[features].fillna(0)

y = df["Target"]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

# ==========================================================
# RANDOM FOREST
# ==========================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(
    X_test
)

print("\nRandom Forest Accuracy")

print(
    accuracy_score(
        y_test,
        rf_pred
    )
)

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# ==========================================================
# XGBOOST
# ==========================================================

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(
    X_train,
    y_train
)

xgb_pred = xgb.predict(
    X_test
)

print("\nXGBoost Accuracy")

print(
    accuracy_score(
        y_test,
        xgb_pred
    )
)

print(
    classification_report(
        y_test,
        xgb_pred
    )
)

prob = xgb.predict_proba(
    X_test
)[:, 1]

print(
    "ROC AUC :",
    roc_auc_score(
        y_test,
        prob
    )
)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance":
    xgb.feature_importances_
})

importance = (
    importance
    .sort_values(
        "Importance",
        ascending=False
    )
)

print("\nFeature Importance")

print(importance)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title(
    "Feature Importance"
)

plt.savefig(
    f"{FIGURE_DIR}/feature_importance.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    xgb,
    f"{MODEL_DIR}/trade_profit_model.pkl"
)

# ==========================================================
# FINAL REPORT
# ==========================================================

report = []

report.append(
    "TRADER PERFORMANCE ANALYSIS REPORT"
)

report.append(
    f"\nTotal Trades : {len(df)}"
)

report.append(
    f"\nAverage PnL : {df['Closed PnL'].mean():.2f}"
)

report.append(
    f"\nWin Rate : {(df['Target'].mean()*100):.2f}%"
)

report.append(
    "\n\nAverage PnL By Sentiment"
)

report.append(
    str(avg_pnl)
)

report.append(
    "\n\nWin Rate By Sentiment"
)

report.append(
    str(win_rate)
)

with open(
    f"{OUTPUT_DIR}/final_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "\n".join(report)
    )

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)

print("\nGenerated Outputs")

print("✔ PnL Distribution Chart")
print("✔ Average PnL Chart")
print("✔ Feature Importance Chart")
print("✔ Top Traders Report")
print("✔ Coin Performance Report")
print("✔ Side Analysis")
print("✔ Direction Analysis")
print("✔ Fee Analysis")
print("✔ T-Test Results")
print("✔ Random Forest Model")
print("✔ XGBoost Model")
print("✔ Final Business Report")