# Loading and Inspecting the Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the original DataCo dataset
df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin1",
    low_memory=False
)

# Inspect the dataset
print("Dataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 5 records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

# Analysing the Quality of the Dataset

# 2.1. Check missing values
print("=== MISSING VALUES ===")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# 2.2. Check duplicate rows
print("\n=== DUPLICATE ROWS ===")
print("Total duplicate rows:", df.duplicated().sum())

# 2.3. Check delivery status
print("\n=== DELIVERY STATUS COUNTS ===")
print(df["Delivery Status"].value_counts())

# 2.4. Check late delivery risk
print("\n=== LATE DELIVERY RISK ===")
print(df["Late_delivery_risk"].value_counts().sort_index())

# 2.5. Check shipping modes
print("\n=== SHIPPING MODE COUNTS ===")
print(df["Shipping Mode"].value_counts())

# 2.6. Check important numerical columns
print("\n=== NUMERICAL SUMMARY ===")
print(
    df[
        [
            "Days for shipping (real)",
            "Days for shipment (scheduled)",
            "Order Item Quantity",
            "Sales",
            "Order Profit Per Order"
        ]
    ].describe().round(2)
)


# STEP 3: EXPLORATORY DATA ANALYSIS

# 3.1. Select important numerical columns
numeric_cols = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount",
    "Order Profit Per Order"
]

# 3.2. Calculate descriptive statistics
print("=== DESCRIPTIVE STATISTICS ===")
print(df[numeric_cols].describe().round(2))

# 3.3. Calculate the difference between actual and scheduled days
df["Shipping_Delay_Days"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

print("\n=== SHIPPING TIME DIFFERENCE ===")
print("Average difference:",
      round(df["Shipping_Delay_Days"].mean(), 2), "days")

print("\nDifference distribution:")
print(df["Shipping_Delay_Days"].value_counts().sort_index())

# 3.4. Summarize shipping performance by shipping mode
print("\n=== PERFORMANCE BY SHIPPING MODE ===")

mode_summary = df.groupby("Shipping Mode").agg(
    Record_Count=("Shipping Mode", "size"),
    Avg_Actual_Days=("Days for shipping (real)", "mean"),
    Avg_Scheduled_Days=("Days for shipment (scheduled)", "mean"),
    Avg_Sales=("Sales", "mean"),
    Avg_Profit=("Order Profit Per Order", "mean"),
    Late_Rate=("Late_delivery_risk", "mean")
)

mode_summary["Late_Rate"] *= 100

print(mode_summary.round(2))

# 3.5. Calculate correlations
print("\n=== CORRELATION MATRIX ===")

correlation = df[numeric_cols].corr()
print(correlation.round(2))

# 3.6. Visualize correlations
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)

plt.title("Correlation Between Logistics Variables")
plt.tight_layout()
plt.show()


# STEP 4: LOGISTICS VISUALIZATIONS

sns.set_theme(style="whitegrid")

# Chart 1: Distribution of actual shipping duration
plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Days for shipping (real)",
    bins=7,
    kde=False
)

plt.title("Distribution of Actual Shipping Duration")
plt.xlabel("Actual Shipping Duration (Days)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()


# Chart 2: Actual vs scheduled shipping duration
plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df.sample(min(3000, len(df)), random_state=42),
    x="Days for shipment (scheduled)",
    y="Days for shipping (real)",
    alpha=0.35
)

plt.title("Actual vs Scheduled Shipping Duration")
plt.xlabel("Scheduled Shipping Duration (Days)")
plt.ylabel("Actual Shipping Duration (Days)")
plt.tight_layout()
plt.show()


# Chart 3: Average actual shipping duration by mode
mode_order = [
    "Same Day",
    "First Class",
    "Second Class",
    "Standard Class"
]

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Shipping Mode",
    y="Days for shipping (real)",
    order=mode_order,
    errorbar=None
)

plt.title("Average Actual Shipping Duration by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Average Actual Shipping Days")
plt.tight_layout()
plt.show()


# STEP 5: OPERATIONAL INSIGHTS

# 5.1. Overall delivery performance
print("=== OVERALL DELIVERY PERFORMANCE ===")

total_records = len(df)
late_records = (df["Late_delivery_risk"] == 1).sum()
late_rate = df["Late_delivery_risk"].mean() * 100

print("Total records:", total_records)
print("Late-risk records:", late_records)
print(f"Late-delivery risk rate: {late_rate:.2f}%")

# 5.2. Delivery status by shipping mode
print("\n=== DELIVERY STATUS BY SHIPPING MODE ===")

status_by_mode = pd.crosstab(
    df["Shipping Mode"],
    df["Delivery Status"],
    normalize="index"
) * 100

print(status_by_mode.round(2))

# 5.3. Shipping performance by region
print("\n=== REGIONAL DELIVERY PERFORMANCE ===")

region_summary = df.groupby("Order Region").agg(
    Record_Count=("Order Region", "size"),
    Late_Records=("Late_delivery_risk", "sum"),
    Average_Actual_Days=("Days for shipping (real)", "mean"),
    Average_Sales=("Sales", "mean")
)

region_summary["Late_Rate_Percent"] = (
    region_summary["Late_Records"]
    / region_summary["Record_Count"] * 100
)

region_summary = region_summary.sort_values(
    "Late_Rate_Percent", ascending=False
)

print(region_summary.round(2).head(10))

# 5.4. Visualize the ten regions with the highest recorded late rate
top_regions = region_summary.head(10).reset_index()

plt.figure(figsize=(11, 6))

sns.barplot(
    data=top_regions,
    x="Late_Rate_Percent",
    y="Order Region",
    errorbar=None
)

plt.title("Top 10 Regions by Recorded Late-Delivery Risk Rate")
plt.xlabel("Late-Delivery Risk Rate (%)")
plt.ylabel("Order Region")
plt.tight_layout()
plt.show()

# 5. Export analytical tables for documentation
mode_summary.to_csv("week3_shipping_mode_summary.csv")
region_summary.to_csv("week3_regional_summary.csv")

print("\nSummary tables exported successfully.")
