"""
Supermarket Sales Exploratory Data Analysis (EDA)
Mini Project - B.Tech 1st Year | Submitted by: Sahil Kumar
Tools: Python, NumPy, Pandas, Matplotlib
Converted from the Google Colab notebook to a plain Python script.
"""

# --- Compatibility shim so the script also runs outside Jupyter/Colab ---
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


# ============================================================
# 1. Import Libraries
# ============================================================
import numpy as np
import pandas as pd

print("Libraries imported successfully!")


# ============================================================
# 2. Upload and Read the Supermarket Dataset
# ============================================================
# Get uploaded file name
file_name = 'SuperMarket Analysis.csv'

# Read using tab separator '\t'
df = pd.read_csv(file_name, sep='\t')

print("Dataset loaded successfully with tab separator!")
print("Rows and columns:", df.shape)
df.head()


# ============================================================
# 3. Understand the Dataset
# ============================================================
print("COLUMN NAMES:")
print(df.columns.tolist())

print("\nDATASET INFORMATION:")
df.info()

print("\nSTATISTICAL SUMMARY:")
df.describe(include="all")


# ============================================================
# 4. Check Missing Values and Duplicate Records
# ============================================================
df.columns = df.columns.str.strip()

print("MISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATE ROWS:", df.duplicated().sum())

df = df.drop_duplicates().copy()
print("Shape after removing duplicates:", df.shape)


# ============================================================
# 5. Clean and Prepare the Data
# ============================================================
numeric_columns = [
    "Unit price", "Quantity", "Tax 5%", "Sales",
    "cogs", "gross margin percentage", "gross income", "Rating"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Map 'Sales' to 'Total' to make subsequent analysis work
if "Sales" in df.columns:
    df["Total"] = df["Sales"]

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("Data preparation completed. 'Total' column mapped from 'Sales'. Available columns:", df.columns.tolist())
df.head()


# ============================================================
# 6. Overall Sales Analysis Using NumPy and Pandas
# ============================================================
if "Total" in df.columns:
    sales = df["Total"].dropna().to_numpy()

    print("TOTAL SALES:", round(np.sum(sales), 2))
    print("AVERAGE BILL:", round(np.mean(sales), 2))
    print("HIGHEST BILL:", round(np.max(sales), 2))
    print("LOWEST BILL:", round(np.min(sales), 2))
    print("NUMBER OF TRANSACTIONS:", len(sales))
else:
    print("The required 'Total' column was not found.")


# ============================================================
# 7. Product Line Analysis
# ============================================================
if {"Product line", "Total"}.issubset(df.columns):
    product_analysis = df.groupby("Product line")["Total"].agg(
        ["sum", "mean", "count"]
    ).sort_values("sum", ascending=False)

    print("SALES BY PRODUCT LINE:")
    display(product_analysis)

    print("TOP PRODUCT LINE:", product_analysis["sum"].idxmax())
else:
    print("Required columns are not available.")


# ============================================================
# 8. Branch-wise Sales Analysis
# ============================================================
if {"Branch", "Total"}.issubset(df.columns):
    branch_analysis = df.groupby("Branch")["Total"].agg(
        ["sum", "mean", "count"]
    ).sort_values("sum", ascending=False)

    print("BRANCH-WISE SALES:")
    display(branch_analysis)

    print("TOP BRANCH:", branch_analysis["sum"].idxmax())
else:
    print("Required columns are not available.")


# ============================================================
# 9. Payment Method Analysis
# ============================================================
if "Payment" in df.columns:
    payment_count = df["Payment"].value_counts()

    print("PAYMENT METHOD COUNTS:")
    display(payment_count)

    print("PAYMENT METHOD PERCENTAGES:")
    display((df["Payment"].value_counts(normalize=True) * 100).round(2))
else:
    print("Payment column was not found.")


# ============================================================
# 10. Customer Type Analysis
# ============================================================
if {"Customer type", "Total"}.issubset(df.columns):
    customer_analysis = df.groupby("Customer type")["Total"].agg(
        ["sum", "mean", "count"]
    )

    print("CUSTOMER TYPE ANALYSIS:")
    display(customer_analysis)
else:
    print("Required columns are not available.")


# ============================================================
# 11. Gender-wise Sales Analysis
# ============================================================
if {"Gender", "Total"}.issubset(df.columns):
    gender_analysis = df.groupby("Gender")["Total"].agg(
        ["sum", "mean", "count"]
    )

    print("GENDER-WISE SALES:")
    display(gender_analysis)
else:
    print("Required columns are not available.")


# ============================================================
# 12. Daily Sales Analysis
# ============================================================
if {"Date", "Total"}.issubset(df.columns):
    daily_sales = (
        df.dropna(subset=["Date"])
          .groupby("Date")["Total"]
          .sum()
          .sort_values(ascending=False)
    )

    print("TOP 10 DAYS BY SALES:")
    display(daily_sales.head(10))

    if not daily_sales.empty:
        print("DATE WITH HIGHEST SALES:", daily_sales.idxmax())
        print("HIGHEST DAILY SALES:", round(daily_sales.max(), 2))
else:
    print("Date or Total column was not found.")


# ============================================================
# 13. Customer Rating Analysis
# ============================================================
if {"Product line", "Rating"}.issubset(df.columns):
    rating_analysis = df.groupby("Product line")["Rating"].mean().sort_values(
        ascending=False
    )

    print("AVERAGE RATING BY PRODUCT LINE:")
    display(rating_analysis.round(2))

    print("HIGHEST RATED PRODUCT LINE:", rating_analysis.idxmax())
else:
    print("Required columns are not available.")


# ============================================================
# 14. Correlation Analysis
# ============================================================
numeric_data = df.select_dtypes(include=np.number)

if numeric_data.shape[1] >= 2:
    print("CORRELATION MATRIX:")
    display(numeric_data.corr().round(2))
else:
    print("Not enough numeric columns for correlation analysis.")


# ============================================================
# 14b. Data Visualizations
# ============================================================
import matplotlib.pyplot as plt

# Configure global plot settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Histogram of Invoice Sales (Total)
axes[0, 0].hist(df['Total'].dropna(), bins=20, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Invoice Totals')
axes[0, 0].set_xlabel('Total Bill ($)')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Payment Method Bar Chart
payment_counts = df['Payment'].value_counts()
axes[0, 1].bar(payment_counts.index, payment_counts.values, color=['coral', 'lightgreen', 'gold'], edgecolor='black')
axes[0, 1].set_title('Transactions by Payment Method')
axes[0, 1].set_xlabel('Payment Method')
axes[0, 1].set_ylabel('Transaction Count')

# Plot 3: Ratings Distribution (Histogram)
axes[1, 0].hist(df['Rating'].dropna(), bins=12, color='plum', edgecolor='black')
axes[1, 0].set_title('Distribution of Customer Ratings')
axes[1, 0].set_xlabel('Rating')
axes[1, 0].set_ylabel('Frequency')

# Plot 4: Top Product Lines by Total Revenue
prod_sales = df.groupby('Product line')['Total'].sum().sort_values()
axes[1, 1].barh(prod_sales.index, prod_sales.values, color='lightgreen', edgecolor='black')
axes[1, 1].set_title('Total Sales by Product Line')
axes[1, 1].set_xlabel('Total Revenue ($)')

plt.tight_layout()
plt.savefig("supermarket_sales_plots.png", dpi=150)
plt.show()


# ============================================================
# 15. Final Project Summary
# ============================================================
print("\nSUPERMARKET SALES EDA SUMMARY")
print("=" * 50)

if "Total" in df.columns:
    print("Total transactions:", df["Total"].count())
    print("Total revenue:", round(df["Total"].sum(), 2))
    print("Average bill:", round(df["Total"].mean(), 2))

if {"Product line", "Total"}.issubset(df.columns):
    best_product = df.groupby("Product line")["Total"].sum().idxmax()
    print("Top product line:", best_product)

if {"Branch", "Total"}.issubset(df.columns):
    best_branch = df.groupby("Branch")["Total"].sum().idxmax()
    print("Top branch:", best_branch)

print("\nPROJECT COMPLETED SUCCESSFULLY!")


# ============================================================
# 16. Save and Download the Final Analysis Report
# ============================================================
df.to_csv("supermarket_sales_eda_report.csv", index=False)

print("CSV report saved successfully!")


# ============================================================
# 17. Conclusion & Key Insights
# ============================================================
