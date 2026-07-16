import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso

def create_visualizations():
    os.makedirs('results/figures', exist_ok=True)
    
    X = np.load('results/X_scaled.npy')
    y = np.load('results/y_scaled.npy')
    
    ls_coeffs = np.load('results/ls_coeffs.npy')
    ridge_coeffs = np.load('results/ridge_coeffs.npy')
    lasso_coeffs = np.load('results/lasso_coeffs.npy')
    
    stability_results = np.load('results/stability_results.npy', allow_pickle=True).item()
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    feature_names = [f'a{i+1}' for i in range(X.shape[1])]
    x = np.arange(len(feature_names))
    width = 0.25
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ls_coeffs_no_bias = ls_coeffs[1:]
    ax1.bar(x - width, ls_coeffs_no_bias, width, label='Least Squares', alpha=0.8)
    ax1.bar(x, ridge_coeffs, width, label='Ridge', alpha=0.8)
    ax1.bar(x + width, lasso_coeffs, width, label='Lasso', alpha=0.8)
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_title('Coefficient Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_names)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('results/figures/coefficient_comparison.png', dpi=300)
    plt.close()
    
    X_with_bias = np.c_[np.ones(X.shape[0]), X]
    y_pred_ls = X_with_bias @ ls_coeffs
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    y_pred_ridge = ridge.predict(X)
    
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X, y)
    y_pred_lasso = lasso.predict(X)
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    methods = [
        ('Least Squares', y_pred_ls, axes2[0]),
        ('Ridge', y_pred_ridge, axes2[1]),
        ('Lasso', y_pred_lasso, axes2[2])
    ]
    for name, y_pred, ax in methods:
        ax.scatter(y, y_pred, alpha=0.6, s=50)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{name}\nR² = {r2_score(y, y_pred):.4f}')
    plt.tight_layout()
    plt.savefig('results/figures/predictions_vs_actual.png', dpi=300)
    plt.close()
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for method in ['least_squares', 'ridge', 'lasso']:
        errors = stability_results[method]
        label = method.replace('_', ' ').title()
        ax3.plot(range(1, len(errors)+1), errors, marker='o', label=label, linewidth=2)
    ax3.set_xlabel('Sample Removed')
    ax3.set_ylabel('MSE')
    ax3.set_title('Stability Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/stability_analysis.png', dpi=300)
    plt.close()
    
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
    residuals_ls = y - y_pred_ls
    residuals_ridge = y - y_pred_ridge
    residuals_lasso = y - y_pred_lasso
    residual_sets = [
        ('Least Squares', residuals_ls, axes4[0]),
        ('Ridge', residuals_ridge, axes4[1]),
        ('Lasso', residuals_lasso, axes4[2])
    ]
    for name, residuals, ax in residual_sets:
        ax.hist(residuals, bins=10, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name}\nStd: {np.std(residuals):.4f}')
    plt.tight_layout()
    plt.savefig('results/figures/error_distribution.png', dpi=300)
    plt.close()
    
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    feature_importance = np.abs(lasso_coeffs)
    indices = np.argsort(feature_importance)[::-1]
    colors = ['red' if importance > 0 else 'blue' for importance in feature_importance[indices]]
    ax5.bar(range(len(feature_importance)), feature_importance[indices], color=colors, alpha=0.7)
    ax5.set_xlabel('Feature Index')
    ax5.set_ylabel('Importance')
    ax5.set_title('Feature Importance (Lasso)')
    ax5.set_xticks(range(len(feature_importance)))
    ax5.set_xticklabels([f'a{i+1}' for i in indices])
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance.png', dpi=300)
    plt.close()
    
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    metrics = ['RMSE', 'MAE', 'R²']
    ls_metrics = [
        np.sqrt(mean_squared_error(y, y_pred_ls)),
        np.mean(np.abs(y - y_pred_ls)),
        r2_score(y, y_pred_ls)
    ]
    ridge_metrics = [
        np.sqrt(mean_squared_error(y, y_pred_ridge)),
        np.mean(np.abs(y - y_pred_ridge)),
        r2_score(y, y_pred_ridge)
    ]
    lasso_metrics = [
        np.sqrt(mean_squared_error(y, y_pred_lasso)),
        np.mean(np.abs(y - y_pred_lasso)),
        r2_score(y, y_pred_lasso)
    ]
    x_metrics = np.arange(len(metrics))
    width_metrics = 0.25
    ax6.bar(x_metrics - width_metrics, ls_metrics, width_metrics, label='Least Squares', alpha=0.8)
    ax6.bar(x_metrics, ridge_metrics, width_metrics, label='Ridge', alpha=0.8)
    ax6.bar(x_metrics + width_metrics, lasso_metrics, width_metrics, label='Lasso', alpha=0.8)
    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Value')
    ax6.set_title('Performance Comparison')
    ax6.set_xticks(x_metrics)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/performance_comparison.png', dpi=300)
    plt.close()
    
    print("Visualizations saved to results/figures/")

if __name__ == "__main__":
    create_visualizations()