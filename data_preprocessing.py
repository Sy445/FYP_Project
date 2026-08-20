# ============================================
# FYP PROJECT
# Predictive Consumer Segmentation and Spending Behaviour in Malaysian Retail
# Final Polished Data Preprocessing Code
# ============================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional: display all columns
pd.set_option("display.max_columns", None)

# --------------------------------------------
# Step 2: Load both CSV files
# --------------------------------------------
df1 = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")
df2 = pd.read_csv("online_retail_.csv", encoding="ISO-8859-1")

print("First dataset shape:", df1.shape)
print("Second dataset shape:", df2.shape)

# --------------------------------------------
# Step 3: Combine both datasets
# --------------------------------------------
df = pd.concat([df1, df2], ignore_index=True)

print("\nCombined dataset shape:")
print(df.shape)

print("\nFirst 5 rows of combined dataset:")
print(df.head())

# --------------------------------------------
# Step 4: Check dataset structure
# --------------------------------------------
print("\nDataset information:")
df.info()

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset shape (rows, columns):")
print(df.shape)

# --------------------------------------------
# Step 5: Summary statistics
# --------------------------------------------
print("\nSummary statistics:")
print(df.describe(include="all"))

# --------------------------------------------
# Step 6: Check missing values
# --------------------------------------------
print("\nMissing values in each column:")
print(df.isnull().sum())

# --------------------------------------------
# Step 7: Remove rows with missing Customer ID
# --------------------------------------------
before_dropna = df.shape[0]
df = df.dropna(subset=["Customer ID"])
after_dropna = df.shape[0]

print(f"\nRows removed due to missing Customer ID: {before_dropna - after_dropna}")
print("Shape after removing missing Customer ID:")
print(df.shape)

# --------------------------------------------
# Step 8: Remove duplicate rows
# --------------------------------------------
before_duplicates = df.shape[0]
df = df.drop_duplicates()
after_duplicates = df.shape[0]

print(f"\nDuplicate rows removed: {before_duplicates - after_duplicates}")
print("Shape after removing duplicates:")
print(df.shape)

# --------------------------------------------
# Step 9: Convert data types
# --------------------------------------------
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")
df["Customer ID"] = df["Customer ID"].astype(str)

# Remove rows with invalid InvoiceDate after conversion
before_date_drop = df.shape[0]
df = df.dropna(subset=["InvoiceDate"])
after_date_drop = df.shape[0]

print(f"\nRows removed due to invalid InvoiceDate: {before_date_drop - after_date_drop}")

print("\nData types after conversion:")
print(df.dtypes)

# --------------------------------------------
# Step 10: Remove invalid transactions
# --------------------------------------------
# Keep only positive Quantity and positive Price
before_filter = df.shape[0]
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
after_filter = df.shape[0]

print(f"\nInvalid rows removed (Quantity <= 0 or Price <= 0): {before_filter - after_filter}")
print("Shape after filtering invalid transactions:")
print(df.shape)

# --------------------------------------------
# Step 11: Create TotalAmount column
# --------------------------------------------
df["TotalAmount"] = df["Quantity"] * df["Price"]

print("\nPreview after creating TotalAmount:")
print(df.head())

# --------------------------------------------
# Step 12: Check outliers using summary statistics
# --------------------------------------------
print("\nSummary statistics for Quantity, Price, and TotalAmount:")
print(df[["Quantity", "Price", "TotalAmount"]].describe())

# --------------------------------------------
# Step 13: Remove outliers using IQR method 
# --------------------------------------------
Q1 = df["TotalAmount"].quantile(0.25)
Q3 = df["TotalAmount"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

before_outlier = df.shape[0]
df = df[(df["TotalAmount"] >= lower_bound) & (df["TotalAmount"] <= upper_bound)]
after_outlier = df.shape[0]

print(f"\nOutliers removed from TotalAmount: {before_outlier - after_outlier}")
print("Shape after removing outliers:")
print(df.shape)

# --------------------------------------------
# Step 14: Final cleaned dataset check
# --------------------------------------------
print("\nFinal dataset information:")
df.info()

print("\nFinal summary statistics:")
print(df.describe(include="all"))

print("\nFinal first 5 rows:")
print(df.head())

# --------------------------------------------
# Step 15: Visualisations
# --------------------------------------------

# Histogram for Quantity
plt.figure(figsize=(8, 5))
plt.hist(df["Quantity"], bins=30, edgecolor="black")
plt.title("Distribution of Quantity")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()

# Histogram for Price
plt.figure(figsize=(8, 5))
plt.hist(df["Price"], bins=30, edgecolor="black")
plt.title("Distribution of Price")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Histogram for TotalAmount
plt.figure(figsize=(8, 5))
plt.hist(df["TotalAmount"], bins=30, edgecolor="black")
plt.title("Distribution of Total Spending")
plt.xlabel("Total Amount")
plt.ylabel("Frequency")
plt.show()

# --------------------------------------------
# Step 16: Save cleaned dataset
# --------------------------------------------
df.to_csv("cleaned_combined_online_retail.csv", index=False)

print("\nCleaned combined dataset has been saved as 'cleaned_combined_online_retail.csv'")