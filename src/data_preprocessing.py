import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import urllib.request
import os

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []

    def download_data(self, data_path='data/'):
        """Download the Adult dataset from UCI repository"""
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        filename = os.path.join(data_path, 'adult.data')

        if not os.path.exists(filename):
            print("Downloading Adult dataset...")
            urllib.request.urlretrieve(url, filename)
            print(f"Dataset downloaded to {filename}")
        else:
            print(f"Dataset already exists at {filename}")

        return filename

    def load_data(self, filepath):
        """Load and return the dataset with proper column names"""
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'educational_num',
            'marital_status', 'occupation', 'relationship', 'race', 'gender',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
        ]

        df = pd.read_csv(filepath, names=column_names, skipinitialspace=True)
        return df

    def clean_data(self, df):
        """Clean the dataset by handling missing values and inconsistencies"""
        # Replace '?' with NaN
        df = df.replace('?', np.nan)

        # Remove rows with missing values
        df = df.dropna()

        # Reset index
        df = df.reset_index(drop=True)

        print(f"Data shape after cleaning: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")

        return df

    def encode_features(self, df, fit=True):
        """Encode categorical variables"""
        df_encoded = df.copy()

        categorical_columns = ['workclass', 'education', 'marital_status', 'occupation',
                             'relationship', 'race', 'gender', 'native_country']

        for col in categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        df_encoded[col] = df_encoded[col].map(lambda x: le.transform([x])[0] 
                                                            if x in le.classes_ else -1)

        # Encode target variable
        if 'income' in df_encoded.columns:
            if fit:
                le_target = LabelEncoder()
                df_encoded['income'] = le_target.fit_transform(df_encoded['income'])
                self.label_encoders['income'] = le_target
            else:
                le_target = self.label_encoders['income']
                df_encoded['income'] = le_target.transform(df_encoded['income'])

        return df_encoded

    def scale_features(self, X_train, X_test=None, fit=True):
        """Scale numerical features"""
        numerical_columns = ['age', 'fnlwgt', 'educational_num', 'capital_gain', 
                           'capital_loss', 'hours_per_week']

        X_train_scaled = X_train.copy()

        if fit:
            X_train_scaled[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
        else:
            X_train_scaled[numerical_columns] = self.scaler.transform(X_train[numerical_columns])

        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_columns] = self.scaler.transform(X_test[numerical_columns])
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def prepare_data(self, test_size=0.2, random_state=42):
        """Complete data preparation pipeline"""
        # Download and load data
        filepath = self.download_data()
        df = self.load_data(filepath)

        print(f"Original data shape: {df.shape}")
        print(f"\nTarget distribution:\n{df['income'].value_counts()}")

        # Clean data
        df_clean = self.clean_data(df)

        # Encode categorical variables
        df_encoded = self.encode_features(df_clean, fit=True)

        # Separate features and target
        X = df_encoded.drop('income', axis=1)
        y = df_encoded['income']

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit=True)

        print(f"\nTraining set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")

        return X_train_scaled, X_test_scaled, y_train, y_test, df_clean

    def preprocess_single_input(self, input_data):
        """Preprocess a single input for prediction"""
        df_input = pd.DataFrame([input_data])

        # Encode categorical variables
        df_encoded = self.encode_features(df_input, fit=False)

        # Scale numerical features
        df_scaled = self.scale_features(df_encoded, fit=False)

        return df_scaled
