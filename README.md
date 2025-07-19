# Employee Income Prediction ML Application

This machine learning project is designed to predict whether an individual's income exceeds **$50K USD per year** based on demographic and work-related attributes. The model is trained on the widely used **UCI Adult Income Dataset** and provides predictions via an intuitive web interface using **Streamlit**.

---

## Project Overview

- **Problem Statement**: Predict the income of individuals (`<=50K` or `>50K`) from structured census and demographic data.
- **Dataset**: UCI Adult Income Census dataset (45K+ instances); includes features like age, education, occupation, capital gains/losses, hours worked per week, etc.
- **Target Variable**: `income` (binary: `<=50K`, `>50K`)


## Features

Train & Compare Multiple ML Models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Linear Regression (for reference)
- Polynomial Regression

Key Functionality:
- Dynamic feature selection via checkboxes
- Model selection to compare performance and predictions
- Automated model evaluation and ranking
- Hyperparameter tuning for selected models (optional)
- Live prediction interface with real-time form inputs
- Optimized training and preprocessing pipeline using `scikit-learn` Pipelines

## Preprocessing Pipeline

The dataset goes through the following preprocessing stages:

1. **Handling Missing Values**: Replaces "?" entries and drops rows with critical missing fields
2. **Label Encoding**: Categorical features are encoded consistently using `LabelEncoder`
3. **Feature Scaling**: Numerical features are scaled using `StandardScaler`
4. **Train-Test Splitting**: 80/20 stratified split using `train_test_split`
5. **Model Pipelines**: Each model is integrated with preprocessing steps using `Pipeline` for reproducibility

## Model Performance Metrics

All models are evaluated using:

- **Accuracy**
- **ROC-AUC**
- **Cross-Validation Score (Mean ± Std Dev)**
- **Classification Report**: Precision, Recall, F1-Score

Best model (based on accuracy): **XGBoost**
- Test Accuracy: 86.16%
- ROC-AUC: 92.04%

## Streamlit Application

A beautiful interactive frontend using Streamlit takes user inputs and delivers real-time income predictions.

### Key UI Features:
- Model selection from dropdown
- Feature selection using checkboxes
- Displays accuracy, AUC, and confusion matrix of selected model
- Upload test rows or manually query by filling form
- Visual themes and responsive design

To launch the app locally:

```
streamlit run app/streamlit_app.py
```


## Project Structure

```
employee-income-ml/
│
├── data/ # raw data or preprocessed CSVs
├── models/ # saved pretrained ML models
├── src/
│ ├── data_preprocessing.py # DataPreprocessor class
│ ├── model_training.py # contains ModelTrainer
│ ├── utils.py # plotting & evaluation utilities
├── app/
│ └── streamlit_app.py # Streamlit UI
├── requirements.txt
├── main_training.py # training entrypoint
├── README.md
```

## 💻 Technologies

- Python 3.10+
- scikit-learn
- XGBoost
- pandas & NumPy
- Streamlit
- seaborn & matplotlib (for EDA & visualization)
- joblib (for model persistence)

## 📈 Sample Output (Model Summary)

| Model              | Accuracy | ROC-AUC | CV Mean | CV Std |
|--------------------|----------|---------|---------|--------|
| XGBoost            | 0.8616   | 0.9204  | 0.8675  | 0.0037 |
| Gradient Boosting  | 0.8589   | 0.9160  | 0.8620  | 0.0033 |
| Random Forest      | 0.8540   | 0.9025  | 0.8542  | 0.0037 |
| Logistic Regression| 0.8163   | 0.8486  | 0.8202  | 0.0045 |
| Decision Tree      | 0.8061   | 0.7398  | 0.8069  | 0.0047 |
| Linear Regression  | 0.7910   | 0.7413  | 0.8001  | 0.0041 |


## Future Enhancements

- SHAP value explanations with bar chart
- Integration with a REST API (FastAPI backend for model inference)

## Author
Project built by Ansh Pradhan as part of the Employee Income Classification ML App

## License
This project is licensed under the MIT License - see the LICENSE file for details
