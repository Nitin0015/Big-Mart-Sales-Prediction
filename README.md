# **Sales Prediction**

This repository contains a Jupyter Notebook that demonstrates how to predict sales for various products across different outlets using machine learning techniques. The dataset used in this project provides information about product attributes, outlet characteristics, and sales data.

---

## **Overview**

Predicting sales is a critical task for businesses to optimize inventory, pricing, and marketing strategies. This project uses an **XGBoost Regressor**, a powerful gradient boosting algorithm, to predict the sales of products based on features such as product type, visibility, MRP (Maximum Retail Price), and outlet characteristics like size, location, and type.

The dataset includes product-level and outlet-level features, with the target variable (`Item_Outlet_Sales`) representing the sales amount for each product at a given outlet.

---

## **Dataset**

- **Source**: The dataset appears to be related to publicly available sales prediction datasets.
- **Features**:
  - `Item_Identifier`: Unique identifier for each product.
  - `Item_Weight`: Weight of the product.
  - `Item_Fat_Content`: Fat content of the product (e.g., Low Fat, Regular).
  - `Item_Visibility`: Visibility of the product in the store.
  - `Item_Type`: Category of the product (e.g., Dairy, Soft Drinks).
  - `Item_MRP`: Maximum Retail Price of the product.
  - `Outlet_Identifier`: Unique identifier for each outlet.
  - `Outlet_Establishment_Year`: Year when the outlet was established.
  - `Outlet_Size`: Size of the outlet (e.g., Small, Medium, High).
  - `Outlet_Location_Type`: Location type of the outlet (e.g., Tier 1, Tier 2).
  - `Outlet_Type`: Type of outlet (e.g., Grocery Store, Supermarket Type1).
- **Target Variable**:
  - `Item_Outlet_Sales`: Sales amount of the product at a given outlet.

---

## **Project Workflow**

1. **Data Loading**:
   - The dataset (`Train.csv`) is loaded into a Pandas DataFrame.
2. **Exploratory Data Analysis (EDA)**:
   - Summary statistics and visualizations are generated using Seaborn and Matplotlib to explore relationships between features and sales.
3. **Data Preprocessing**:
   - Missing values are handled appropriately.
   - Categorical variables are encoded into numerical values using Label Encoding.
4. **Model Training**:
   - An XGBoost Regressor is trained to predict sales amounts.
   - The dataset is split into training and testing sets using `train_test_split`.
5. **Model Evaluation**:
   - Performance metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are calculated to evaluate model accuracy.

---

## **Dependencies**

To run this project, you need the following Python libraries:

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost

You can install these dependencies using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost
```

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/SalesPrediction.git
   cd SalesPrediction
   ```

2. Ensure that the dataset file (`Train.csv`) is in the same directory as the notebook.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Sales-Pred.ipynb
   ```

4. Run all cells in the notebook to execute the code.

---

## **Results**

The XGBoost Regressor provides predictions for sales amounts based on input features. Evaluation metrics such as MAE and RMSE indicate how well the model performs in predicting sales. Further improvements can be made by experimenting with hyperparameter tuning or feature engineering.

---

## **Acknowledgments**

- The dataset was sourced from publicly available sales prediction datasets or competitions.
- Special thanks to XGBoost developers for providing a robust machine learning library.

---
