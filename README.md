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

### **📝 Split Data into Features (X) and Target (y)**

[In 8]:

- The dataset was split into **features (X)** and **target (y)**, where:
  - **X** contains all the independent variables (features), and
  - **y** contains the target variable `Churn`.

```python
# Split the data into features (X) and target (y)
x = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']  # Target

# Split into training and testing sets (70/30 split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

[In 9]:

###📝 **Standardize the Features Using StandardScaler**
- The features were standardized using **StandardScaler** to ensure that all features have a mean of 0 and a standard deviation of 1. This step is important because many machine learning algorithms perform better when the features are on the same scale.
- The **training set** was fitted and transformed, while the **test set** was only transformed (to avoid data leakage).

```python
# Standardize the features using StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

### 📝 **Apply Model - Random Forest Classifier**

- **Random Forest Classifier** was trained on the scaled features.
- The model was evaluated using accuracy on both the **training** and **test** sets.

```python
# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train_scaled, y_train)

# Make predictions on training and test sets
y_pred_train = clf.predict(x_train_scaled)
y_pred_test = clf.predict(x_test_scaled)

# Evaluate model accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
train_balanced_acc = balanced_accuracy_score(y_train, y_pred_train)
test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)

# Print the results
print(f'Training Accuracy: {train_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Training Balanced Accuracy: {train_balanced_acc:.4f}')
print(f'Test Balanced Accuracy: {test_balanced_acc:.4f}')
```
### **💡Conclusion: ** 

The evaluation of the model's performance on both the training and test sets is as follows:

- **Training Accuracy:** 1.0000  
  - The model achieves perfect accuracy on the training set. However, this could indicate overfitting, as the model has learned the training data very well.
  
- **Test Accuracy:** 0.9343  
  - The model performs well on the test set, with a high accuracy of 93.43%. This suggests that the model generalizes well to unseen data, although there is a slight drop from the training accuracy, which is expected.

- **Training Balanced Accuracy:** 1.0000  
  - The training balanced accuracy is perfect (1.0000), indicating that the model is equally good at predicting both classes (Churn and Non-Churn) in the training set.

- **Test Balanced Accuracy:** 0.8421  
  - The balanced accuracy on the test set is 84.21%, which is still quite good. This metric takes into account both classes and gives a better sense of the model's performance in imbalanced datasets.

-> The model performs well overall, with **high accuracy** and **balanced accuracy** on both the training and test sets. The **perfect training accuracy** indicates a **risk of overfitting**, but the **test accuracy** and **balanced accuracy** show that the model still **generalizes well to new data**. It is essential to **monitor the model on additional datasets** to confirm its **robustness in real-world scenarios**.

### 📝 **Apply Random Forest To Find Important Features**

[In 10]:

```python
# Get feature importances from the best model
feats = {feature: importance for feature, importance in zip(x_train.columns, best_clf.feature_importances_)}

# Create a DataFrame for feature importances
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importances'})
importances = importances.sort_values(by='Importances', ascending=True).reset_index()

# Plot the feature importances
plt.figure(figsize=(10,10))
plt.barh(importances.tail(20)['index'], importances.tail(20)['Importances'])  # Plot top 20 features
plt.title('Feature Importance')
plt.show()
```

[Out 10]:

![image](https://github.com/user-attachments/assets/6216d1d9-fc0c-43b0-a40b-f3647378a36d)

### **💡Conclusion: ** 

Base on the image, we can conclude that top 5 important features hwich directly affecting to churn behaviors are: Tenture, CashbackAmount, WarehouseTohome, Complain, DaySinceLastOrder.
These features play a crucial role in predicting whether a customer is likely to churn. Tenure and Days Since Last Order indicate customer engagement, while Cashback Amount and Complaints reflect satisfaction levels. Warehouse to Home Distance may influence delivery experience, impacting customer retention. 
Then, we will plot a histogram chart to visualize the differences between churn and non-churn behavior for the top important features. This will help us identify patterns and understand how these features contribute to customer churn. 

### 📝 **Plot Histogram**

![Image](https://github.com/user-attachments/assets/2179d426-4175-407c-bc82-a8972f3a1de0)

## 4️⃣ **Key Findings and Recommendations for Retention**  

| Metric                     | Churn (Blue) | Non-Churn (Yellow) | Insight |
|----------------------------|-------------|---------------------|---------|
| **Tenure (Customer Lifespan)** | 80% leave within 5 months, very few stay beyond 10 months | More evenly distributed, many customers stay over 20 months | **Churned customers tend to leave very early**, meaning the initial experience is crucial. Without an effective retention strategy, they will churn quickly. |
| **CashbackAmount (Cashback Received)** | Average around 100-200, widely distributed | Mostly concentrated between 120-250 | **Churned customers receive less cashback**, which may lead to a lower perceived financial benefit. Increasing cashback or introducing alternative incentives might help retention. |
| **WarehouseToHome (Delivery Time)** | Wide distribution, average around 15-30 days, some exceeding 35 days | Mostly under 20 days, rarely exceeding 25 days | **Longer delivery times correlate with higher churn rates**. Optimizing logistics and reducing shipping time could enhance customer retention. |
| **Complain (Customer Complaints)** | Two extremes: either 0 or 1, with ~50%-50% lodging complaints | Similar pattern, but most customers do not complain, complains only accoounts for 10-15% | **Churned customers have a higher complaint rate**, but many also leave without complaining. This suggests that some dissatisfied customers simply churn without providing feedback, requiring a more proactive approach. |
| **DaySinceLastOrder (Days Since Last Order)** | Widely spread, most over 10 days, with many exceeding 20 days | Mostly under 10 days, rarely over 15 days | **Churned customers order less frequently**, meaning if they don’t make a purchase soon, they are likely to leave. A strategy to encourage repeat purchases, such as discounts for quick reorders, could be beneficial. |

1. **The initial experience is a critical factor** → If customers are dissatisfied early on, they are likely to churn.  
2. **Lower cashback may reduce retention rates** → Consider adjusting cashback incentives or introducing alternative rewards.  
3. **Longer delivery times increase churn risk** → Optimize shipping logistics to shorten delivery times.  
4. **Many customers churn without lodging complaints** → Proactively gather feedback before they decide to leave.  
5. **Low order frequency is an early churn indicator** → Implement strategies like time-sensitive discounts to encourage repeat purchases sooner.

   
