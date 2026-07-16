import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

def ridge_regression_solution():
    X = np.load('results/X_scaled.npy')
    y = np.load('results/y_scaled.npy')
    
    print("Ridge Regression Method")
   
    
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_score = -np.inf
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    print(f"Best alpha: {best_alpha}")
    
    start_time = time.time()
    
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X, y)
    
    end_time = time.time()
    
    y_pred = ridge.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R^2: {r2:.6f}")
    print(f"Time: {end_time - start_time:.6f}s")
    
    np.save('results/ridge_coeffs.npy', ridge.coef_)
    
    return ridge

if __name__ == "__main__":
    ridge_regression_solution()