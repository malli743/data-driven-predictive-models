# 📊 Telco Customer Churn Prediction

This project aims to predict whether a customer will churn (i.e., discontinue their telecom service) based on their account information, service usage, and demographic details. It uses supervised machine learning techniques to classify customers into churn or non-churn categories.

## 🚀 Objective

To build a reliable machine learning model that can identify customers likely to leave the service, enabling businesses to take proactive retention measures.

## 🧠 Problem Type

**Binary Classification**  
Target Variable: `Churn` (Yes/No → 1/0)

---

## 📁 Dataset

**Source**: Telco Customer Churn dataset  
**File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
**Features Include**:
- Demographics (e.g., gender, senior citizen)
- Account info (e.g., tenure, MonthlyCharges, TotalCharges)
- Services (e.g., InternetService, Contract, PaymentMethod)
- Target: `Churn` (Yes or No)

---

## 🛠️ Project Workflow

### 1. Load the Data
```python
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
````

Import the dataset to start exploration and preprocessing.

### 2. Data Cleaning

* Convert `TotalCharges` to numeric
* Handle missing values using median
* Drop irrelevant columns like `customerID`

### 3. Encode Categorical Variables

* Binary encode the target (`Churn`)
* One-hot encode other categorical columns

```python
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)
```

### 4. Exploratory Data Analysis (EDA)

* Visualize churn distribution and its relationship with key features
* Plot correlation heatmaps for insights

### 5. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```

Standardize numeric features like `MonthlyCharges`, `tenure` to bring them to a common scale.

### 6. Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(...)
```

Split the data to evaluate model generalization.

### 7. Model Training

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

Train a Random Forest model to classify churn effectively.

### 8. Feature Importance

```python
import matplotlib.pyplot as plt
import pandas as pd

pd.Series(model.feature_importances_, index=X_train.columns).nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.show()
```

### 9. Model Tuning (Optional)

Use `GridSearchCV` to optimize hyperparameters and improve performance.

### 10. Save the Model

```python
import joblib
joblib.dump(model, 'telco_churn_model.pkl')
```

---

## 🎯 Business Value

By identifying potential churners early, businesses can:

* Offer personalized retention incentives
* Improve customer service interactions
* Reduce customer loss and increase revenue

---

## 🧰 Tech Stack

* **Python**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**
* **Scikit-learn**
* **Joblib**

---

## ✅ Output

A trained and serialized model (`telco_churn_model.pkl`) ready for deployment, with insights into which features most impact customer churn.

---

## 📌 Future Improvements

* Add support for real-time predictions via API
* Integrate into a dashboard for business users
* Test other models like XGBoost or SVM for performance comparison

---

## 📎 License

This project is open-source and available under the MIT License.

