import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load data
df = pd.read_csv(r"E:\ML\data.csv")

# --- VERY IMPORTANT STEP FOR CONFUSION MATRIX ---
# A Confusion Matrix ONLY works for Classification problems (categories like Yes/No, High/Low).
# Because your target variable 'E' has continuous decimal numbers (12.0, 12.8, etc.), 
# we must convert it into categories to generate a confusion matrix.
# Here, we categorize 'E' into 'High' (1) and 'Low' (0) based on its median value.
median_E = df['E'].median()
df['E_class'] = (df['E'] >= median_E).astype(int)

# 1. Heatmap (Correlation)
plt.figure(figsize=(8, 6))
# Dropping the new categorical class and ID for the pure correlation
sns.heatmap(df.drop(['sample_number', 'E_class'], axis=1).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 2. Outliers (Boxplot)
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['A', 'B', 'C', 'D']])
plt.title('Outliers in Features (Boxplot)')
plt.show()

# --- MODEL TRAINING ---
# Features and New Categorical Target
X = df[['A', 'B', 'C', 'D']]
y = df['E_class']

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest CLASSIFIER (used for confusion matrix instead of Regressor)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Class (0)', 'High Class (1)'], 
            yticklabels=['Low Class (0)', 'High Class (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()

# 4. Feature Importance
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Random Forest Feature Importance')
plt.show()
plt.close()

print(f"Target 'E' Median Used for Splitting: {median_E}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))