# Credit Default Prediction Using XGBoost

## **Project Overview**
This project focuses on building a machine learning model using **XGBoost** to predict whether a borrower will default on their loan. We preprocess raw financial and behavioral data, engineer features, and optimize an XGBoost classifier to maximize predictive performance.

**Final AUC Score: 0.84**

---

## **1. Data Processing & Cleaning**

### **1.1 Data Loading**
Three main datasets were loaded:
- **`PPD_LogInfo_3_1_Training_Set.csv`**: Contains user login information.
- **`PPD_Training_Master_GBK_3_1_Training_Set.csv`**: The main dataset with loan application details.
- **`PPD_Userupdate_Info_3_1_Training_Set.csv`**: Tracks user profile updates over time.

### **1.2 Train-Test Split**
- The dataset was split into **70% training data** and **30% testing data** based on unique user IDs (`Idx`).

---

## **2. Feature Engineering**
Feature extraction focused on **behavioral and historical attributes** to capture loan applicants' financial stability.

### **2.1 Handling Login Information**
- **Time-based Features**: We extracted login frequencies within time windows **(7, 30, 60, 90, 120, 150, 180 days before application)**.
- **Diversity Metrics**: We calculated unique login methods used within each window.
- **Activity Intensity**: Average login count per unique login method.

### **2.2 User Profile Update Analysis**
- Standardized categorical values (`MOBILEPHONE` → `PHONE`, `IDNumber` → `IDNUMBER`).
- Calculated update frequencies within different time windows.
- Tracked critical updates (ID number, marriage status, phone, car ownership) to identify high-risk behaviors.

### **2.3 Missing Value Handling**
- If a feature had **>80% missing values**, it was removed.
- Numerical missing values were **randomly imputed** using observed values from the same feature.

### **2.4 Outlier Detection & Scaling**
- Used **Interquartile Range (IQR)** to detect outliers.
- **Min-max scaling** was applied to normalize numerical features.

---

## **3. Feature Selection**

### **3.1 Removing Imbalanced Features**
- Identified categorical features where **one value appeared in ≥90% of records**.
- Compared **bad sample rates (default rates)** between majority and minority classes.
- If the minority class did not show a significantly higher bad sample rate, the feature was removed.

### **3.2 Recursive Feature Elimination (RFE)**
- Used **XGBoost feature importance** to iteratively drop unimportant variables.
- Reduced features from **400 to 158** while maintaining performance.

---

## **4. Model Training & Hyperparameter Tuning**

### **4.1 Baseline Model**
- Trained an **XGBoost Classifier** with default hyperparameters.
- Initial AUC Score: **0.78**.

### **4.2 Hyperparameter Tuning (GridSearchCV)**
To maximize AUC, we tuned the following parameters:

#### **Step 1: Optimizing Tree Structure**
```python
param_test1 = {'max_depth': range(3, 10, 2), 'min_child_weight': range(1, 6, 2)}
```
- **Best values:** `max_depth=9`, `min_child_weight=3`.

#### **Step 2: Adjusting Split Gain**
```python
param_test2 = {'gamma': [i/10.0 for i in range(0, 5)]}
```
- **Best value:** `gamma=0` (no split penalty required).

#### **Step 3: Feature Subsampling**
```python
param_test3 = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
```
- **Best values:** `subsample=0.6`, `colsample_bytree=0.8`.

#### **Step 4: Regularization (L1 Penalty)**
```python
param_test4 = {'reg_alpha': [0.01, 0.1, 1, 10, 50, 100, 200, 500]}
```
- **Best value:** `reg_alpha=50`.

#### **Step 5: Optimizing Number of Trees**
```python
param_test5 = {'n_estimators': range(100, 401, 10)}
```
- **Best value:** `n_estimators=390`.

---

## **5. Model Evaluation**

### **5.1 Final Model Performance**
- **AUC Score: 0.84** ✅
- **ROC Curve** shows good separation between positive (default) and negative (non-default) classes.

### **5.2 Feature Importance Analysis**
- The top predictors included **login frequency trends, recent profile updates, and financial indicators**.
- Some highly imbalanced features were dropped, as they did not contribute to AUC improvement.

---

## **6. Key Takeaways**
### ✅ **What Worked Well?**
- **Feature engineering** significantly improved performance (behavioral & time-based features were valuable).
- **Hyperparameter tuning** improve AUC from **0.78 → 0.84**.
- **Removing redundant features** reduced complexity (reduce from 402 features to 128) while maintaining accuracy as 0.84.

### ❌ **Challenges & Areas for Improvement**
- **Handling rare categorical values:** Some features with rare values might still have predictive power.
- **More advanced imputation methods:** Using deep learning or advanced statistical techniques for missing values.
- **Further model comparison:** Trying LightGBM or CatBoost to check for further improvements.

---

## **7. Conclusion**
This project successfully built a **high-performance credit default prediction model** with an AUC of **0.84**. The approach combined **strong feature engineering, careful data preprocessing, and advanced hyperparameter tuning**. These techniques can be applied to real-world credit risk models to improve financial decision-making.

🚀 **Next Steps:**
- Deploy the model in a **real-time credit scoring system**.
- Perform **model monitoring** to ensure accuracy over time.


📌 **For further improvements, we can consider testing ensemble methods or deep learning approaches!** 🚀

