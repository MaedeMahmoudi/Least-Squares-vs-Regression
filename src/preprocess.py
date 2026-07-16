import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

def load_and_preprocess():
    df = pd.read_csv('data/dataset.csv')
    X = df.drop('b', axis=1).values
    y = df['b'].values
    
    X[:, 4] = np.log1p(X[:, 4])
    X[:, 5] = np.log1p(X[:, 5])
    X[:, 6] = np.log1p(X[:, 6])
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    os.makedirs('results', exist_ok=True)
    np.save('results/X_scaled.npy', X_scaled)
    np.save('results/y_scaled.npy', y_scaled)
    joblib.dump(scaler_X, 'results/scaler_X.pkl')
    joblib.dump(scaler_y, 'results/scaler_y.pkl')
    
    print("Preprocessing completed")
    print(f"X shape: {X_scaled.shape}")
    print(f"y shape: {y_scaled.shape}")
    
    return X_scaled, y_scaled

if __name__ == "__main__":
    load_and_preprocess()