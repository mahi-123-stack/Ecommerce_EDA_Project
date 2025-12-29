print("python file is working")
import pandas as pd

df = pd.read_csv("Amazon Sale Report.csv",low_memory=False)

print(df.shape)
print(df.head())

print("\n---Dataset Information---")
print(df.info())

print("\n---Statistical Summary---")
print(df.describe())

print("\n---Column Name---")
print(df.columns)

#checking missing values col-wise
print("\n---Missing Values Count:\n")
print(df.isnull().sum())

print("\n---Data Types---\n")
print(df.dtypes)

#fill missing numeric value with median
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

#fill categorical value with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

#verify the missing values now
print("\n-------Missing values after cleaning---------\n")
df.isnull().sum()

#to check corelation analysis(numeric relationships)
numeric_df = df.select_dtypes(include=['int64', 'float64'])
print("\nCorrelation matrix:")
print(numeric_df.corr())

#imp : basic bussiness insights
#orders per category
if 'Category' in df.columns:
    print("\nOrders per Category:")
    print(df['Category'].value_counts())


#orders per status
if 'Status' in df.columns:
    print("\nOrders per Order Status:")
    print(df['Status'].value_counts())

#save cleaned data(very imp.)
df.to_csv("Amazon_Sales_Cleaned.csv", index=False)
print("Cleaned dataset saved successfully.")

#at this stage, industry-standard EDA : Data visualization(most imp for EDA)...[EDA phase 2]
#matplotlib = base plotting
#seaborn = cleaner statistical plots
#whitegrid = readable charts


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df.columns=df.columns.str.strip().str.lower()

#ORDER STATUS DISTRIBUTION..
# ---------------- VISUALIZATION ---------------- #

# 1️⃣ Sales Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['amount'],kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales Amount")
plt.ylabel("Frequency")
plt.show()

# 2️⃣ Orders by Status
plt.figure(figsize=(7,4))
df['status'].value_counts().plot(kind='bar')
plt.title("Orders by Status")
plt.xlabel("Order Status")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.show()

# 3️⃣ Orders by Sales Channel
plt.figure(figsize=(6,4))
df['sales channel'].value_counts().plot(kind='bar')
plt.title("Orders by Sales Channel")
plt.xlabel("Sales Channel")
plt.ylabel("Count")
plt.show()

# 4️⃣ Top 10 Product Categories
plt.figure(figsize=(7,4))
df['category'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Product Categories")
plt.xlabel("Category")
plt.ylabel("Orders")
plt.xticks(rotation=30)
plt.show()

# 5️⃣ Top 10 Products (SKU-based)
plt.figure(figsize=(7,4))
df['sku'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Most Ordered Products")
plt.xlabel("SKU")
plt.ylabel("Orders")
plt.xticks(rotation=30)
plt.show()

#Bussiness-Driven Analysis(level-up)
#Step 1-----KPI METRICS-----(numbers that matter)
# ---- KPI METRICS ----
total_orders = len(df)
total_sales = df['amount'].sum()
avg_order_value = df['amount'].mean()
print("Total Orders:", total_orders)
print("Total Sales:", round(total_sales, 2))
print("Average Order Value:", round(avg_order_value, 2))

#Step 2----STATUS vs SALES (core business insight)
#this will tell do cancelled/returned orders hurt revenue??
# ---- Sales by Order Status ----
status_sales = df.groupby('status')['amount'].sum().sort_values(ascending=False)
plt.figure(figsize=(7,4))
status_sales.plot(kind='bar')
plt.title("Total Sales by Order Status")
plt.xlabel("Order Status")
plt.ylabel("Total Sales")
plt.xticks(rotation=30)
plt.show()

#Step 3----TOP REVENUE CATEGORIES(not just count)
#count != revenue.....now we analyze money
# ---- Revenue by Category ----
category_sales = df.groupby('category')['amount'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(7,4))
category_sales.plot(kind='bar')
plt.title("Top 10 Categories by Revenue")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
plt.show()

#Step 4---Write Insights---(most imp)
"""
INSIGHTS:
1. Majority of revenue comes from delivered orders.
2. Cancelled and returned orders contribute negligible sales.
3. Amazon.in dominates as the primary sales channel.
4. A small number of product categories drive most revenue.
5. Sales distribution is right-skewed (few high-value orders).
"""

#Correlation Analysis (Final Tech step)
#does order quantity  or price affect revenue??
# ---- Correlation Analysis ----
numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(6,4))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#Saving final Clean Dataset(mandatory for github and real-world projects)
# ---- Save Cleaned Dataset ----
df.to_csv("Final_Cleaned_Amazon_Sales.csv", index=False)
print("Final cleaned dataset saved successfully.")


