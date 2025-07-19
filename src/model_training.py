import sys
import os
import joblib
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from utils import generate_eda_report, create_model_comparison_chart
import pandas as pd

def main():
    """Main training pipeline"""
    print("Starting Employee Income Prediction ML Pipeline")
    print("="*60)

    # Step 1: Data Preprocessing
    print("\nSTEP 1: Data Preprocessing")
    print("-" * 30)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, df_clean = preprocessor.prepare_data()

    # Generate EDA report
    eda_report = generate_eda_report(df_clean)
    print(f"\nDataset Overview:")
    print(f"Shape: {eda_report['shape']}")
    print(f"Features: {list(X_train.columns)}")

    # Step 2: Model Training
    print("\nSTEP 2: Model Training")
    print("-" * 30)
    trainer = ModelTrainer()
    trainer.train_models(X_train, y_train, X_test, y_test)

    # Display model summary
    summary_df = trainer.get_model_summary()

    # Step 3: Hyperparameter Tuning (Optional)
    print("\nSTEP 3: Hyperparameter Tuning")
    print("-" * 40)

    # Tune the best performing model
    if trainer.best_model_name in ['Random Forest', 'XGBoost']:
        print(f"Tuning {trainer.best_model_name}...")
        tuned_model = trainer.hyperparameter_tuning(X_train, y_train, trainer.best_model_name)

        # Evaluate tuned model
        tuned_results = trainer.evaluate_model(tuned_model, X_test, y_test, f"Tuned {trainer.best_model_name}")
        print(f"\nTuned model accuracy: {tuned_results['accuracy']:.4f}")

        # Update best model if tuned version is better
        if tuned_results['accuracy'] > trainer.model_results[trainer.best_model_name]['accuracy']:
            trainer.best_model = tuned_model
            print("Tuned model performs better. Updated best model.")
    else:
        print(f"Hyperparameter tuning skipped for {trainer.best_model_name}")

    # Step 4: Feature Importance Analysis
    print("\nSTEP 4: Feature Importance Analysis")
    print("-" * 40)
    if hasattr(trainer.best_model, 'feature_importances_'):
        importance_df = trainer.plot_feature_importance(trainer.best_model, preprocessor.feature_names)
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

    # Step 5: Final Evaluation
    print("\nSTEP 5: Final Model Evaluation")
    print("-" * 40)
    final_results = trainer.evaluate_model(trainer.best_model, X_test, y_test, trainer.best_model_name)

    # Step 6: Save Models and Preprocessor
    print("\nSTEP 6: Saving Models")
    print("-" * 30)

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Save best model
    trainer.save_model(trainer.best_model, trainer.best_model_name)

    # Save preprocessor
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Preprocessor saved to models/preprocessor.pkl")

    # Save model metadata
    metadata = {
        'best_model_name': trainer.best_model_name,
        'best_accuracy': trainer.model_results[trainer.best_model_name]['accuracy'],
        'feature_names': preprocessor.feature_names,
        'model_results': {k: {
            'accuracy': v['accuracy'],
            'roc_auc': v['roc_auc'],
            'cv_mean': v['cv_mean']
        } for k, v in trainer.model_results.items()}
    }

    import json
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Model metadata saved to models/model_metadata.json")

    # Final Summary
    print("\nTRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Best Model: {trainer.best_model_name}")
    print(f"Test Accuracy: {trainer.model_results[trainer.best_model_name]['accuracy']:.4f}")
    print(f"ROC-AUC Score: {trainer.model_results[trainer.best_model_name]['roc_auc']:.4f}")
    print("\nFiles created:")
    print("- models/preprocessor.pkl")
    print(f"- models/{trainer.best_model_name.lower().replace(' ', '_')}_model.pkl")
    print("- models/model_metadata.json")
    print("\nReady for deployment!")

if __name__ == "__main__":
    main()
