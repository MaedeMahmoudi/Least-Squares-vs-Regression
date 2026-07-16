import numpy as np
import time
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def lasso_regression_solution():
    X = np.load('results/X_scaled.npy')
    y = np.load('results/y_scaled.npy')
    
    print("Lasso Regression Method")
    
    
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_score = -np.inf
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        scores = cross_val_score(lasso, X, y, cv=5, scoring='r2')
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    print(f"Best alpha: {best_alpha}")
    
    start_time = time.time()
    
    lasso = Lasso(alpha=best_alpha, max_iter=10000)
    lasso.fit(X, y)
    
    end_time = time.time()
    
    y_pred = lasso.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    n_selected = np.sum(np.abs(lasso.coef_) > 1e-6)
    
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R^2: {r2:.6f}")
    print(f"Time: {end_time - start_time:.6f}s")
    print(f"Selected features: {n_selected}")
    
    np.save('results/lasso_coeffs.npy', lasso.coef_)
    
    return lasso

if __name__ == "__main__":
    lasso_regression_solution()