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

1️⃣ **Exploratory Data Analysis (EDA)**  

2️⃣ **Train & Apply Churn Prediction Model**  

3️⃣ **Key Findings and Recommendations for Retention**  

4️⃣ **Customer Segmentation By Clustering**  

5️⃣ **Insights & Recommendations**

