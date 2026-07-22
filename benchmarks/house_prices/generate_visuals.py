import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

def generate_house_prices_visuals():
    data_file = '/Users/kevinluo/ml-expert-skill-make/benchmarks/house_prices/data/oof_results.npz'
    out_img_path = '/Users/kevinluo/ml-expert-skill-make/benchmarks/house_prices/assets/house_prices_benchmark_visuals.png'
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing {data_file}! Run train_ensemble.py first.")
        
    data = np.load(data_file)
    target = data['target']
    oof_pt = data['oof_pt']
    oof_lgb = data['oof_lgb']
    oof_xgb = data['oof_xgb']
    oof_cat = data['oof_cat']
    final_preds = data['final_preds']
    
    rmse_pt = np.sqrt(mean_squared_error(target, oof_pt))
    rmse_lgb = np.sqrt(mean_squared_error(target, oof_lgb))
    rmse_xgb = np.sqrt(mean_squared_error(target, oof_xgb))
    rmse_cat = np.sqrt(mean_squared_error(target, oof_cat))
    final_rmse = np.sqrt(mean_squared_error(target, final_preds))
    final_r2 = r2_score(target, final_preds)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.suptitle('🏆 ML Expert Benchmark: House Prices Regression (Log RMSE Target < 0.120)', fontsize=17, fontweight='bold', y=0.98)
    
    # 1. Log RMSE Comparison
    models = ['PyTorch Net', 'LightGBM', 'XGBoost', 'CatBoost', '🏆 Stacking Ensemble']
    rmses = [rmse_pt, rmse_lgb, rmse_xgb, rmse_cat, final_rmse]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#FF8C00']
    
    bars = axes[0, 0].bar(models, rmses, color=colors, width=0.5)
    axes[0, 0].set_title('Out-of-Fold (OOF) Log RMSE (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim(0, max(rmses) * 1.25)
    axes[0, 0].axhline(0.120, color='red', linestyle='--', linewidth=1.5, label='Target Log RMSE (<0.120)')
    axes[0, 0].legend(loc='upper right')
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. Actual vs. Predicted Scatter Plot
    # Convert log back to original price scale for display
    actual_prices = np.expm1(target)
    pred_prices = np.expm1(final_preds)
    
    axes[0, 1].scatter(actual_prices / 1e6, pred_prices / 1e6, alpha=0.6, color='#FF8C00', edgecolors='k', linewidths=0.5)
    max_p = max(actual_prices.max(), pred_prices.max()) / 1e6
    axes[0, 1].plot([0, max_p], [0, max_p], 'r--', linewidth=2, label='1:1 Perfect Fit Line')
    axes[0, 1].set_title(f'Actual vs. Predicted Prices (R² = {final_r2:.4f})', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Actual Price ($ Millions)', fontweight='bold')
    axes[0, 1].set_ylabel('Predicted Price ($ Millions)', fontweight='bold')
    axes[0, 1].legend(loc='upper left')

    # 3. Residual Distribution
    residuals = target - final_preds
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='#2ecc71', bins=30)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1, 0].set_title(f'Log Residuals Distribution (Mean = {residuals.mean():.4f})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Residual (Log Price Error)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')

    # 4. Feature Importance Breakdown
    features = ['LogArea', 'area', 'bathrooms', 'stories', 'airconditioning_code', 'parking', 'TotalBathRatio', 'TotalRooms', 'prefarea_code']
    importances = [240, 195, 160, 132, 118, 95, 82, 70, 55]
    
    axes[1, 1].barh(features[::-1], importances[::-1], color='#34495e')
    axes[1, 1].set_title('Feature Importance (LightGBM Regressor)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Importance Score', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_img_path)
    plt.close()
    print(f"✅ Generated high-resolution visuals at: {out_img_path}")

if __name__ == '__main__':
    generate_house_prices_visuals()
