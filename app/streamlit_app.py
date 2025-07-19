import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from utils import format_prediction_output, prepare_streamlit_input_mappings
import os

# Page configuration
st.set_page_config(
    page_title="Employee Income Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    try:
        # Load the best model (assuming it's saved)
        model_files = [f for f in os.listdir('models') if f.endswith('_model.pkl')]
        if not model_files:
            st.error("No trained model found. Please run main_training.py first.")
            return None, None, None

        # Load metadata to get the best model name
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)

        best_model_file = f"models/{metadata['best_model_name'].lower().replace(' ', '_')}_model.pkl"
        model = joblib.load(best_model_file)
        preprocessor = joblib.load('models/preprocessor.pkl')

        return model, preprocessor, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def create_input_form():
    """Create the input form for user data"""
    st.sidebar.header("Enter Employee Information")

    # Get input mappings
    mappings = prepare_streamlit_input_mappings()

    # Demographic Information
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age", 17, 90, 39)
    gender = st.sidebar.selectbox("Gender", mappings['gender'])
    race = st.sidebar.selectbox("Race", mappings['race'])
    native_country = st.sidebar.selectbox("Native Country", mappings['native_country'])

    # Education Information
    st.sidebar.subheader("Education")
    education = st.sidebar.selectbox("Education Level", mappings['education'])
    educational_num = st.sidebar.slider("Education Years", 1, 16, 10)

    # Work Information
    st.sidebar.subheader("Work Details")
    workclass = st.sidebar.selectbox("Work Class", mappings['workclass'])
    occupation = st.sidebar.selectbox("Occupation", mappings['occupation'])
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

    # Family Information
    st.sidebar.subheader("Family")
    marital_status = st.sidebar.selectbox("Marital Status", mappings['marital_status'])
    relationship = st.sidebar.selectbox("Relationship", mappings['relationship'])

    # Financial Information
    st.sidebar.subheader("Financial")
    fnlwgt = st.sidebar.number_input("Final Weight", 12285, 1484705, 189778)
    capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
    capital_loss = st.sidebar.number_input("Capital Loss", 0, 4356, 0)

    # Compile input data
    input_data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'educational_num': educational_num,
        'marital_status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'hours_per_week': hours_per_week,
        'native_country': native_country
    }

    return input_data

def display_prediction_results(prediction_output):
    """Display prediction results with visualizations"""
    col1, col2 = st.columns([1, 1])

    with col1:
        # Prediction result
        if prediction_output['prediction'] == '>50K':
            st.success(f"Predicted Income: **{prediction_output['prediction']}**")
        else:
            st.info(f"Predicted Income: **{prediction_output['prediction']}**")

        st.write(f"**Confidence:** {prediction_output['confidence']:.1f}%")

    with col2:
        # Probability breakdown
        prob_data = prediction_output['probability_breakdown']
        fig = go.Figure(data=[
            go.Bar(
                x=list(prob_data.keys()),
                y=list(prob_data.values()),
                marker_color=['lightcoral' if k == '<=50K' else 'lightgreen' for k in prob_data.keys()]
            )
        ])

        fig.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Income Category",
            yaxis_title="Probability (%)",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

def display_model_info(metadata):
    """Display information about the trained model"""
    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best Model", metadata['best_model_name'])

    with col2:
        st.metric("Test Accuracy", f"{metadata['best_accuracy']:.3f}")

    with col3:
        st.metric("Total Features", len(metadata['feature_names']))

    # Model comparison chart
    if len(metadata['model_results']) > 1:
        st.subheader("Model Comparison")

        models = list(metadata['model_results'].keys())
        accuracies = [metadata['model_results'][model]['accuracy'] for model in models]
        roc_aucs = [metadata['model_results'][model]['roc_auc'] for model in models]

        comparison_df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracies,
            'ROC-AUC': roc_aucs
        })

        fig = px.bar(comparison_df, x='Model', y=['Accuracy', 'ROC-AUC'], 
                    title="Model Performance Comparison", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    # Load model and preprocessor
    model, preprocessor, metadata = load_model_and_preprocessor()

    if model is None:
        st.stop()

    # App title and description
    st.title("Employee Income Predictor")
    st.markdown("""
    This application predicts whether an employee's income is above or below $50K based on demographic and work-related features.

    **How to use:**
    1. Fill in the employee information in the sidebar
    2. Click 'Predict Income' to get the prediction
    3. View the results and model confidence
    """)

    # Create input form
    input_data = create_input_form()

    # Prediction button
    if st.sidebar.button("🔮 Predict Income", type="primary"):
        try:
            # Preprocess input data
            processed_input = preprocessor.preprocess_single_input(input_data)

            # Make prediction
            prediction = model.predict(processed_input)
            probability = model.predict_proba(processed_input)

            # Format output
            prediction_output = format_prediction_output(prediction, probability)

            # Display results
            st.subheader("Prediction Results")
            display_prediction_results(prediction_output)

            # Display input summary
            st.subheader("Input Summary")
            input_df = pd.DataFrame([input_data]).T
            input_df.columns = ['Value']
            st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    # Display model information
    st.markdown("---")
    display_model_info(metadata)

    # Footer
    st.markdown("---")
    st.markdown("**Note:** This model is trained on the Adult Census Income dataset and should be used for educational purposes only.")

if __name__ == "__main__":
    main()
