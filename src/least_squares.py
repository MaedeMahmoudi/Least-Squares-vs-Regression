import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def least_squares_solution():
    X = np.load('results/X_scaled.npy')
    y = np.load('results/y_scaled.npy')
    
    print("Least Squares Method")
  
    
    start_time = time.time()
    
    X_with_bias = np.c_[np.ones(X.shape[0]), X]
    coeffs = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    
    end_time = time.time()
    
    y_pred = X_with_bias @ coeffs
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R^2: {r2:.6f}")
    print(f"Time: {end_time - start_time:.6f}s")
    
    np.save('results/ls_coeffs.npy', coeffs)
    
    return coeffs

if __name__ == "__main__":
    least_squares_solution()