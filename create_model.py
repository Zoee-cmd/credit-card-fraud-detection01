# create_model.py its purpose is to create a ML model for fraud detection and saves files for dashboard to use.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Generate synthetic data for training
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate features
    data = {
        'amount': np.random.lognormal(mean=5, sigma=1, size=n_samples),
        'transaction_type': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        'hour': np.random.randint(0, 24, size=n_samples),
        'day_of_week': np.random.randint(0, 7, size=n_samples),
        'location_risk': np.random.uniform(0, 1, size=n_samples),
        'device_risk': np.random.uniform(0, 1, size=n_samples),
        'time_since_last_transaction': np.random.exponential(1, size=n_samples),
        'avg_transaction_amount': np.random.lognormal(mean=5, sigma=0.5, size=n_samples),
        'transaction_frequency': np.random.poisson(lam=5, size=n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target (fraud or not)
    # Higher probability of fraud for:
    # - Large amounts
    # - Unusual hours
    # - High risk locations/devices
    fraud_score = (
        0.3 * (df['amount'] > df['amount'].quantile(0.95)).astype(int) +
        0.2 * ((df['hour'] < 6) | (df['hour'] > 22)).astype(int) +
        0.25 * (df['location_risk'] > 0.8).astype(int) +
        0.25 * (df['device_risk'] > 0.8).astype(int)
    )
    
    df['is_fraud'] = (fraud_score > 0.5).astype(int)
    
    return df

# Generate data
print("Generating synthetic data...")
df = generate_synthetic_data()

# Split features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the scaler
print("Creating and fitting scaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create and train the model
print("Training fraud detection model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
print("Saving model and scaler...")
with open('models/fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model files created successfully!")
print("You can now run the fraud_dashboard.py script.") 