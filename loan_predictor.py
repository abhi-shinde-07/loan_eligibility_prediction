import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sqlite3
import pickle
import os

class LoanPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.db_path = 'loan_applications.db'
        self.model_path = 'loan_model.pkl'
        self.scaler_path = 'scaler.pkl'
        self.initialize_db()
        
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.load_model()
        else:
            self.train_model()
    
    def initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                applicant_name TEXT,
                gender TEXT,
                married TEXT,
                dependents INTEGER,
                education TEXT,
                self_employed TEXT,
                applicant_income REAL,
                coapplicant_income REAL,
                loan_amount REAL,
                loan_term INTEGER,
                credit_history INTEGER,
                property_area TEXT,
                prediction_result INTEGER,
                prediction_proba REAL,
                submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def train_model(self):
        # Load sample dataset (in a real scenario, this would be your historical loan data)
        data = pd.read_csv('https://raw.githubusercontent.com/awsaf49/loan-eligibility/master/data/loan-train.csv')
        
        # Preprocessing
        data = data.drop(['Loan_ID'], axis=1)
        data['Dependents'] = data['Dependents'].replace('3+', 3).astype(float)
        
        # Handle missing values
        for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
            data[col] = data[col].fillna(data[col].mode()[0])
        
        # Convert categorical to numerical
        data = pd.get_dummies(data, drop_first=True)
        
        # Split data
        X = data.drop('Loan_Status_Y', axis=1)
        y = data['Loan_Status_Y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        
        # Save model and scaler
        pickle.dump(self.model, open(self.model_path, 'wb'))
        pickle.dump(self.scaler, open(self.scaler_path, 'wb'))
    
    def load_model(self):
        self.model = pickle.load(open(self.model_path, 'rb'))
        self.scaler = pickle.load(open(self.scaler_path, 'rb'))
    
    def preprocess_input(self, input_data):
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Convert categorical to numerical (matching training data format)
        df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
        
        # One-hot encoding (ensure all expected columns are present)
        df = pd.get_dummies(df)
        
        # Ensure all columns from training are present
        expected_columns = [
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Credit_History', 'Dependents', 'Gender_Male', 'Married_Yes', 
            'Education_Not Graduate', 'Self_Employed_Yes', 'Property_Area_Semiurban', 
            'Property_Area_Urban'
        ]
        
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training data
        df = df[expected_columns]
        
        # Scale features
        scaled_data = self.scaler.transform(df)
        return scaled_data
    
    def predict(self, input_data):
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            proba = self.model.predict_proba(processed_data)[0][1]
            
            return prediction, proba
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None
    
    def save_application(self, applicant_data, prediction, probability):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO applications (
                applicant_name, gender, married, dependents, education, 
                self_employed, applicant_income, coapplicant_income, 
                loan_amount, loan_term, credit_history, property_area, 
                prediction_result, prediction_proba
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            applicant_data.get('applicant_name', ''),
            applicant_data['gender'],
            applicant_data['married'],
            applicant_data['dependents'],
            applicant_data['education'],
            applicant_data['self_employed'],
            applicant_data['applicant_income'],
            applicant_data['coapplicant_income'],
            applicant_data['loan_amount'],
            applicant_data['loan_term'],
            applicant_data['credit_history'],
            applicant_data['property_area'],
            int(prediction),
            float(probability)
        ))
        
        conn.commit()
        conn.close()
    
    def get_application_history(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql('SELECT * FROM applications ORDER BY submission_date DESC', conn)
        conn.close()
        return df