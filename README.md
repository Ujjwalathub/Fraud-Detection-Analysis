🚦 Getting Started
Prerequisites
Make sure you have Python installed along with the required libraries:

Bash
pip install pandas numpy matplotlib seaborn scikit-learnRunning the Analysis
Clone this repository to your local machine.

Ensure the data/ directory contains all 10 raw CSV files.

Execute the main analysis script:

Bash
python data.py
This will automatically merge the datasets, output statistical summaries to the console, and generate visualization images (.png) in your working directory.

📁 Project Structure
Plaintext
├── data/
│   ├── account_activity.csv
│   ├── amount_data.csv
│   ├── anomaly_scores.csv
│   ├── customer_data.csv
│   ├── fraud_indicators.csv
│   ├── merchant_data.csv
│   ├── suspicious_activity.csv
│   ├── transaction_category_labels.csv
│   ├── transaction_metadata.csv
│   └── transaction_records.csv
├── visualizations/
│   ├── age_fraud.png                 # Age distribution by fraud class
│   ├── fraud_rate_category.png       # Risk levels per transaction type
│   ├── correlation_heatmap.png       # Feature correlation matrix
│   └── confusion_matrix.png          # Model performance visualization
├── merged_dataset.csv                # The auto-generated consolidated dataset
├── data.py                           # Main execution script
└── README.md                         # This documentation file
🔬 Methodology
Data Ingestion & Integration: * Merged transaction-level data on TransactionID.

Merged customer-level data on CustomerID.

Joined the final dataset with merchant information using MerchantID.

Exploratory Data Analysis (EDA):

Analyzed statistical distributions of amounts, account balances, and ages.

Grouped fraud rates by categorical variables (Category, SuspiciousFlag).

Generated correlation heatmaps to identify linear relationships.

Data Preprocessing:

Isolated numerical features (Amount, TransactionAmount, AnomalyScore, Age, AccountBalance) for modeling.

Baseline Modeling:

Split data into 80% Training and 20% Testing sets.

Trained a RandomForestClassifier with 100 estimators.

Evaluated using Confusion Matrix, Accuracy Score, and a detailed Classification Report.

📊 Key Findings & EDA
The "Suspicious" Flag Matters: Customers who were previously flagged with SuspiciousFlag = 1 had a drastically higher fraud rate (12.0%) compared to standard customers (4.3%).

Category Risk: Online transactions are the highest risk vector (5.10% fraud rate), closely followed by Other and Retail. Travel transactions showed the lowest risk (3.53%).

Demographics: The average age of a customer caught in a fraudulent transaction (38.5 years) was slightly lower than the average legitimate customer (39.9 years).

Anomaly Scores are Insufficient: The pre-calculated AnomalyScore showed a very weak correlation (-0.048) with the actual FraudIndicator, proving that heuristic rules alone cannot reliably detect fraud in this environment.

🤖 Model Evaluation (The Accuracy Paradox)
Our baseline Random Forest model yielded the following results on the test set (200 transactions):

Overall Accuracy: 96.0%

Precision (Fraud): 0.00

Recall (Fraud): 0.00

F1-Score (Fraud): 0.00

Interpretation
At first glance, 96% accuracy appears excellent. However, looking at the Confusion Matrix, the model predicted 0 (Legitimate) for all 200 test cases. Because there were only 7 actual fraud cases in the test set, the model only got 7 wrong, resulting in 96% accuracy. It completely failed to identify a single fraudulent transaction.

Feature Importance
Despite failing to predict the minority class, the model ranked the importance of the features it attempted to use:

TransactionAmount (~24.1%)

AnomalyScore (~22.1%)

AccountBalance (~20.3%)

Amount (~19.6%)

Age (~13.7%)

🚀 Future Work & Next Steps
To transform this baseline into a production-ready fraud detection system, the severe class imbalance must be handled. The following roadmap is planned:

1. Advanced Resampling Techniques
SMOTE (Synthetic Minority Over-sampling Technique): Generate synthetic examples of the minority (fraud) class during the training phase so the model can learn its distinct patterns.

ADASYN / Undersampling: Combine SMOTE with Tomek Links or random undersampling of the majority class to balance the dataset.

2. Algorithmic Adjustments
Cost-Sensitive Learning: Utilize the class_weight='balanced' parameter more aggressively or implement custom penalty matrices to heavily penalize false negatives (missing a fraud case).

Gradient Boosting: Transition from Random Forest to XGBoost or LightGBM, which often handle imbalanced tabular data more effectively through iterative boosting.

3. Metric Optimization
Transition the primary evaluation metric from Accuracy to Recall (catching as many frauds as possible), F1-Score, and the Area Under the Precision-Recall Curve (PR AUC).
