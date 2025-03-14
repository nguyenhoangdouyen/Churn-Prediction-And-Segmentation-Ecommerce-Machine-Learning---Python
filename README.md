# **Churn Prediction & Segmentation For Retention Strategy | Ecommerce | Machine Learning - Python**


**Author:** Nguyễn Hoàng Đỗ UYên

**Date:** March 2025

**Tools Used:** Machine Learning - Python 


## 📑 Table of Contents

📌 Background & Overview

📂 Dataset Description & Data Structure

🔎 Final Conclusion & Recommendations


## 📌 Background & Overview

### **🎯 Objective**

This project focuses on **predicting and segmenting churned users** in an e-commerce business to **develop effective retention strategies**. By leveraging **Machine Learning & Python**, this project aims to:

✔️ Identify key behaviors and patterns of churned users.

✔️ Develop a predictive model to forecast customer churn.

✔️ Segment churned users to personalize retention offers and promotions.


**❓ What Business Question Will It Solve?**

✔️ What factors contribute to customer churn in e-commerce?

✔️ How can we predict churned users and take proactive measures?

✔️ How to build an accurate churn prediction model?

✔️ How can we segment churned users for targeted promotions?


**👤 Who Is This Project For?**

✔️ Data Analysts & Business Analysts – To gain insights into churn behavior and retention strategies.

✔️ Marketing & Customer Retention Teams – To design data-driven promotional campaigns.

✔️ Decision-makers & Stakeholders – To reduce churn and improve customer lifetime value.


## 📂 **Dataset Description & Data Structure**

### 📌 **Data Source**  
**Source:** The dataset is obtained from the e-commerce company's database.  
**Size:** The dataset contains 5,630 rows and 20 columns.  
**Format:** .xlxs file format.

### 📊 **Data Structure & Relationships**

1️⃣ **Tables Used:**  
The dataset contains only **1 table** with customer and transaction-related data.

2️⃣ **Table Schema & Data Snapshot**  
**Table: Customer Churn Data**

<details>
  <summary>Click to expand the table schema</summary>

| **Column Name**              | **Data Type** | **Description**                                              |
|------------------------------|---------------|--------------------------------------------------------------|
| CustomerID                   | INT           | Unique identifier for each customer                          |
| Churn                        | INT           | Churn flag (1 if customer churned, 0 if active)              |
| Tenure                       | FLOAT         | Duration of customer's relationship with the company (months)|
| PreferredLoginDevice         | OBJECT        | Device used for login (e.g., Mobile, Desktop)                 |
| CityTier                     | INT           | City tier (1: Tier 1, 2: Tier 2, 3: Tier 3)                   |
| WarehouseToHome              | FLOAT         | Distance between warehouse and customer's home (km)         |
| PreferredPaymentMode         | OBJECT        | Payment method preferred by customer (e.g., Credit Card)     |
| Gender                       | OBJECT        | Gender of the customer (e.g., Male, Female)                  |
| HourSpendOnApp               | FLOAT         | Hours spent on app or website in the past month              |
| NumberOfDeviceRegistered     | INT           | Number of devices registered under the customer's account   |
| PreferedOrderCat             | OBJECT        | Preferred order category for the customer (e.g., Electronics)|
| SatisfactionScore            | INT           | Satisfaction rating given by the customer                    |
| MaritalStatus                | OBJECT        | Marital status of the customer (e.g., Single, Married)       |
| NumberOfAddress              | INT           | Number of addresses registered by the customer               |
| Complain                     | INT           | Indicator if the customer made a complaint (1 = Yes)         |
| OrderAmountHikeFromLastYear  | FLOAT         | Percentage increase in order amount compared to last year   |
| CouponUsed                   | FLOAT         | Number of coupons used by the customer last month            |
| OrderCount                   | FLOAT         | Number of orders placed by the customer last month           |
| DaySinceLastOrder            | FLOAT         | Days since the last order was placed by the customer        |
| CashbackAmount               | FLOAT         | Average cashback received by the customer in the past month  |

</details>

## **⚒️ Main Process

## 1️⃣ **Data Preprocessing**  

📌 Import Necessary Libraries

[In 1]: 

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score, 
    precision_score, f1_score, confusion_matrix, silhouette_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
```
📂 Mount Google Drive to Access Files

[In 2]: 

```python
# Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

# Define the path to the project folder
path = '/content/drive/MyDrive/ML_Final Project_Nguyen Hoang Do Uyen/'

# Load the data
df = pd.read_excel(path + 'churn_prediction.xlsx')
```

📂 Before diving into analysis, let's take a quick look at the first few rows of the dataset to examine its structure and key features

[In 3]:

```python
df.head(5)
```

[Out 3]:

![Image](https://github.com/user-attachments/assets/c79b4dee-2ffe-4deb-a9d5-9f2052465f45)

#### **💡 Data Understanding**

📌 Before performing any analysis or modeling, I carried out several steps to preprocess the data:

**📝 Checked Dataset Structure**  

After checking the general structure of the dataset, this gave me an overview of the number of rows, columns, and data types for each feature, along with summary statistics.

  - The dataset contains 5,630 rows and 20 columns, with a mix of numeric and categorical variables.
  - Missing values were identified in several columns, such as `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, etc.

**📝 Checked for Missing Values**  
Missing values were detected in multiple columns. The columns with missing values are:

   - `Tenure` - 264 missing values
   - `WarehouseToHome` - 251 missing values
   - `HourSpendOnApp` - 255 missing values
   - `OrderAmountHikeFromlastYear` - 265 missing values
   - `CouponUsed` - 256 missing values
   - `OrderCount` - 258 missing values
   - `DaySinceLastOrder` - 307 missing values

**📝 Checked for Duplicates**  

Aftering checkeing for duplicate rows in the dataset and found that there were no duplicate entries.

### **💡 Summary** 

- The dataset contained missing values in several columns. The missing values were **handled by replacing them with the mean**, which prepared the data for further analysis and modeling.
- The dataset contains words with the **same meaning** but written differently. These should be **standardized into a single form**.

**📝 Missing Value Handling**  

[In 4]:

```python
# Define the list of columns with missing values
cols_missing = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']

# Replace missing columns with median
for col in cols_missing:
    # Fill missing values in each column with the median of that column
    df[col].fillna(value= df[col].median(), inplace=True)
```

**📝 Merging Columns with Similar Data but Different Names**  

[In 5]:

```python
# Merge columns with the same data but different names
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
    'COD': 'Cash on Delivery',  # Replace 'COD' with 'Cash on Delivery'
    'CC': 'Credit Card'  # Replace 'CC' with 'Credit Card'
})
```

## 2️⃣ **Exploratory Data Analysis (EDA)**

**📝 Explored Continuous Variables**  

Explored the distribution of continuous variables to better understand their uniqueness and spread. Most of the continuous variables had a limited number of unique values, but this is reasonable given the context of the dataset.

**📝 Univariate Analysis**  
- Categorical variables like `PreferredLoginDevice`, `Gender`, `MaritalStatus`, and others were analyzed through count plots, which gave insights into the distribution of each category.
  
- Continuous variables like `Tenure`, `SatisfactionScore`, `CashbackAmount`, and others were examined using boxplots. It was observed that the outliers in these columns. However, **outliers** reasonable and **should be kept** because they represent **distinguishing characteristics for predicting churn**.


**📝 Final Data Inspection**

[In 6]:
```python
# Print dataset info to verify missing values and column types
print(df.info())
```

[Out 6]:

![Image](https://github.com/user-attachments/assets/616fb912-8efd-4d42-a3a6-df95e4550e6a)

## 3️⃣ **Train & Apply Churn Prediction Model**

### **📝 Encoding**

After preprocessing the dataset, encoding was applied to the categorical features:

1. **One-Hot Encoding**:
   - The columns with categorical features having a small number of unique values were encoded using **one-hot encoding**. This creates binary columns for each unique value, making it easier for the model to process categorical data.
   - The columns encoded are:
     - `PreferredLoginDevice`
     - `PreferredPaymentMode`
     - `PreferedOrderCat`
     - `MaritalStatus`

2. **Label Encoding**:
   - The `Gender` column was encoded using **label encoding** to convert categorical labels into numerical values (0 or 1).
   
3. **Dropped Unnecessary Column**:
   - The `CustomerID` column was dropped since it is a unique identifier and does not contribute to the prediction model.

[In 7]:

```python
# Apply one-hot encoding to categorical columns with a small number of unique values
df_encoded = pd.get_dummies(df, columns=['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus', 'PreferredLoginDevice'])

# Apply label encoding to the 'Gender' column (converts categorical labels to numerical values)
label_encoder = LabelEncoder()
df_encoded['Gender'] = label_encoder.fit_transform(df_encoded['Gender'])

# Drop the 'CustomerID'
df_encoded = df_encoded.drop(columns=['CustomerID'])
```

[Out 7]:

![Image](https://github.com/user-attachments/assets/9fade8d4-b571-41de-a8be-40dc78952773)
