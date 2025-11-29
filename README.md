# Shinkansen Travel Experience – Passenger Satisfaction Prediction 🚄

This repository contains machine learning solution for the **Shinkansen Travel Experience** hackathon, where the goal was to predict whether a passenger was **satisfied (1)** or **not satisfied (0)** with their overall experience on the Shinkansen bullet train.

> 🏆 **Best Score Achieved:** Accuracy = **0.9590192** on the Hacakaton leaderboard 
> 📊 **Model:** CatBoostClassifier with 5-Fold Stratified Cross-Validation

---

## 1. Problem Overview

The objective is to build a classification model that predicts the **`Overall_Experience`** of passengers based on:

- **Survey data** – service quality ratings (e.g., Catering, Cleanliness, Seat Comfort, Online Support, etc.)
- **Travel data** – demographic and journey features (e.g., Age, Customer_Type, Travel_Class, Travel_Distance, Delays)

The final submission is a CSV with the following columns:

- `ID`
- `Overall_Experience` (0 = Not Satisfied, 1 = Satisfied)

---

## 2. Data Description

Two datasets were provided, each split into **train** and **test**:

- **Surveydata_train / Surveydata_test**
  - `ID`
  - `Overall_Experience` (train only)
  - Categorical service ratings (Catering, Seat_Comfort, Cleanliness, Onboard_Service, etc.)

- **Traveldata_train / Traveldata_test**
  - `ID`
  - `Gender`, `Customer_Type`, `Type_Travel`, `Travel_Class`
  - `Age`, `Travel_Distance`
  - `Departure_Delay_in_Mins`, `Arrival_Delay_in_Mins`

The final training set is obtained by **merging survey and travel data on `ID`**.

> **Note:** Original datasets are not included in this repository due to hackathon usage terms. Use your own copies in the `data/` folder.

---

## 3. Approach & Methodology

### 3.1 Data Preparation

- Merged `survey_train` and `travel_train` on `ID` to create a consolidated **training dataset**.
- Merged `survey_test` and `travel_test` on `ID` to create a consolidated **test dataset**.
- Defined:
  - Features: all columns except `ID` and `Overall_Experience`
  - Target: `Overall_Experience`
- Identified:
  - **Categorical features** by `dtype == object`
  - **Numerical features** by non-object dtypes

### 3.2 Missing Value Handling

- **Categorical features**:
  - Converted to string and imputed missing values with a placeholder category `"missing"`.
- **Numerical features**:
  - Imputed missing values using the **median** of each feature from the training data.

This ensured no missing values remained before modeling.

---

## 4. Modeling

### 4.1 Algorithm

The final solution uses **CatBoostClassifier**, which handles categorical features natively.

Key parameters (CPU mode):

```python
cat_params = dict(
    iterations=6000,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=10,
    loss_function="Logloss",
    eval_metric="Accuracy",
    bootstrap_type="Bayesian",
    bagging_temperature=0.5,
    rsm=0.6,
    one_hot_max_size=12,
    random_strength=1.5,
    leaf_estimation_iterations=8,
    border_count=254,
    od_type="Iter",
    od_wait=200,
    task_type="CPU",
    verbose=200
)
