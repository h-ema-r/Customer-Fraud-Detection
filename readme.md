# Fraud Detection in E-commerce Transactions

Given codes performs a series of steps to build and evaluate machine learning model  and deep learning model for predicting fraudulent transactions using customer data . It includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation. 

## Overview of Script

### 1. Importing Libraries and Data
The code starts by importing necessary Python libraries such as pandas, numpy, seaborn, and matplotlib for data manipulation and visualization. It then loads two datasets: one containing customer details (`customers_df.csv`) and another containing transaction details (`customer_transaction_details.csv`).

### 2. Initial Data Exploration
Basic information about the datasets is displayed, including the number of rows and columns, data types of the columns, and some basic statistics. This helps in understanding the structure and characteristics of the data.

### 3. Data Cleaning
Unnecessary columns (like 'Unnamed: 0', presumably added during data export) are removed from both datasets.

### 4. Data Aggregation and Merging
The code aggregates transaction data by customer email to get the total transaction amount and the number of failed transactions per customer. This aggregated data is then merged with the customer details to form a single dataset that includes both customer information and transaction metrics.

## Merged Data Columns
The `merged_data` dataframe contains the merged information from both the customer details and the aggregated transaction data.The columns of the `merged_data` dataframe include:
- customerEmail
- customerPhone
- customerDevice
- customerIPAddress
- customerBillingAddress
- No_Transactions
- No_Orders
- No_Payments
- Total_Transaction_Amount
- Total_Failed_Transactions
- Fraud

### 5. Feature Engineering
The merged dataset is restructured to place the target variable 'Fraud' at the end for easier access. Duplicate records based on 'customerEmail' are identified and examined.

### 6. Exploratory Data Analysis (EDA)
Various visualizations are created to explore the relationships within the data:
- Histograms of numerical features to understand their distributions.
- Bar plots to visualize the count of fraudulent versus non-fraudulent transactions.
- Box plots to examine the distribution of numerical features across fraudulent and non-fraudulent transactions.
- A heatmap to visualize the correlation matrix of numerical features, which can provide insights into potential multicollinearity.

### 7. Data Preprocessing
- Removal of additional non-informative columns such as 'customerEmail' and other identifying information that are not useful for modeling.
- Missing values in 'transactionAmount' and 'transactionFailed' are imputed using the median of each column.
- Numerical features are scaled using StandardScaler to normalize their ranges.

## Machine Learning model

### 8. Model Preparation and Evaluation
- A range of classifiers including Logistic Regression, Decision Tree, Random Forest, SVM, and Gradient Boosting are initialized.
- The target variable 'Fraud' is oversampled using RandomOverSampler to address class imbalance in the dataset.
- The dataset is split into training and test subsets.
- Each model is trained and evaluated using cross-validation on the training data to assess generalizability.
- Final evaluation is performed on the test set to obtain accuracy scores for each model.

### 9. Model Training and Testing
Each classifier is trained on the resampled training set, and predictions are made on the test set. The accuracy of each model is calculated and printed, providing a direct comparison of their performance on the task of fraud detection.


## Fraud Detection Deep Learning Model

This contains code for building and evaluating a deep learning model for fraud detection in e-commerce transactions. The script preprocesses the data by defining features and target variables, addressing class imbalance using Random Over-Sampling (ROS), and splitting the data into training and testing sets. The model, constructed using Keras Sequential API, includes densely connected layers with ReLU activation functions and dropout regularization to prevent overfitting. It is trained using binary cross-entropy loss and the Adam optimizer, with training history visualized to monitor performance. Finally, the model is evaluated on test data to assess its accuracy in detecting fraudulent transactions.


## CONCLUSION

In conclusion, the deep learning model, achieved the highest accuracy score of approximately 90.69% on the test data, showcasing its effectiveness in fraud detection. However, Random Forest also performed impressively, achieving an accuracy score of approximately 88.37%. These results highlight the importance of exploring various machine learning algorithms and techniques to address the task of fraud detection in e-commerce transactions.

