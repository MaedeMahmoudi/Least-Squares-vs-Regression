import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

def stability_analysis():
    X = np.load('results/X_scaled.npy')
    y = np.load('results/y_scaled.npy')
    
    ls_coeffs = np.load('results/ls_coeffs.npy')
    
    ridge_alpha = 1.0
    lasso_alpha = 0.01
    
    n_samples = X.shape[0]
    results = {
        'least_squares': [],
        'ridge': [],
        'lasso': []
    }
    
    for i in range(min(10, n_samples)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]
        y_test = y[i:i+1]
        
        X_with_bias = np.c_[np.ones(X_train.shape[0]), X_train]
        coeffs_ls = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y_train
        X_test_with_bias = np.c_[np.ones(1), X_test]
        y_pred_ls = X_test_with_bias @ coeffs_ls
        
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        
        lasso = Lasso(alpha=lasso_alpha, max_iter=10000)
        lasso.fit(X_train, y_train)
        y_pred_lasso = lasso.predict(X_test)
        
        results['least_squares'].append(mean_squared_error(y_test, y_pred_ls))
        results['ridge'].append(mean_squared_error(y_test, y_pred_ridge))
        results['lasso'].append(mean_squared_error(y_test, y_pred_lasso))
    
    print("Stability Analysis Results:")
    print("-" * 40)
    
    for method in results:
        errors = results[method]
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        print(f"{method}: Mean={mean_err:.6f}, Std={std_err:.6f}")
    
    np.save('results/stability_results.npy', results)
    
    return results

if __name__ == "__main__":
    stability_analysis()