import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_confusion_matrix(y_true, y_pred, labels=['<=50K', '>50K'], title='Confusion Matrix'):
    """Plot confusion matrix using seaborn"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    return plt

def plot_feature_distribution(df, feature, target='income', figsize=(12, 5)):
    """Plot feature distribution by target variable"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Overall distribution
    if df[feature].dtype == 'object':
        df[feature].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title(f'Distribution of {feature}')
        ax1.tick_params(axis='x', rotation=45)

        # Distribution by target
        pd.crosstab(df[feature], df[target], normalize='index').plot(kind='bar', ax=ax2)
        ax2.set_title(f'{feature} by {target}')
        ax2.tick_params(axis='x', rotation=45)
    else:
        df[feature].hist(bins=30, ax=ax1)
        ax1.set_title(f'Distribution of {feature}')

        # Box plot by target
        df.boxplot(column=feature, by=target, ax=ax2)
        ax2.set_title(f'{feature} by {target}')

    plt.tight_layout()
    return fig

def create_model_comparison_chart(model_results):
    """Create interactive comparison chart for model performance"""
    models = list(model_results.keys())
    accuracies = [model_results[model]['accuracy'] for model in models]
    roc_aucs = [model_results[model]['roc_auc'] for model in models]

    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=models, y=accuracies, marker_color='lightblue'),
        go.Bar(name='ROC-AUC', x=models, y=roc_aucs, marker_color='lightcoral')
    ])

    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500
    )

    return fig

def generate_eda_report(df):
    """Generate basic EDA statistics"""
    report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict(),
        'categorical_summary': {}
    }

    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        report['categorical_summary'][col] = df[col].value_counts().to_dict()

    return report

def prepare_streamlit_input_mappings():
    """Prepare mappings for Streamlit input widgets"""
    mappings = {
        'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                     'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
        'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                     'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
                     '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
        'marital_status': ['Married-civ-spouse', 'Divorced', 'Never-married', 
                          'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
        'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                      'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                      'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                      'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
        'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                        'Other-relative', 'Unmarried'],
        'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
        'gender': ['Female', 'Male'],
        'native_country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 
                          'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 
                          'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 
                          'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 
                          'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 
                          'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 
                          'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 
                          'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    }
    return mappings

def format_prediction_output(prediction, probability):
    """Format prediction output for display"""
    income_labels = ['<=50K', '>50K']
    predicted_class = income_labels[prediction[0]]
    confidence = probability[0][prediction[0]] * 100

    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probability_breakdown': {
            income_labels[0]: probability[0][0] * 100,
            income_labels[1]: probability[0][1] * 100
        }
    }
