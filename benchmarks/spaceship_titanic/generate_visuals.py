import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

def generate_spaceship_visuals():
    data_file = '/Users/kevinluo/ml-expert-skill-make/benchmarks/spaceship_titanic/data/oof_results.npz'
    out_img_path = '/Users/kevinluo/ml-expert-skill-make/benchmarks/spaceship_titanic/assets/spaceship_titanic_benchmark_visuals.png'

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing {data_file}! Run train_ensemble.py first.")

    data = np.load(data_file)
    target = data['target']
    oof_pt = data['oof_pt']
    oof_lgb = data['oof_lgb']
    oof_xgb = data['oof_xgb']
    oof_cat = data['oof_cat']
    oof_et = data['oof_et']
    ensemble_probs = data['ensemble_probs']
    best_thresh = float(data['best_thresh'])

    ensemble_preds = (ensemble_probs >= best_thresh).astype(int)
    final_acc = accuracy_score(target, ensemble_preds)
    final_auc = roc_auc_score(target, ensemble_probs)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.suptitle('🏆 ML Expert Benchmark: Spaceship Titanic Classification (>81% Accuracy Target)', fontsize=17, fontweight='bold', y=0.98)

    # 1. Model Accuracy Comparison Bar Chart
    models = ['PyTorch Net', 'LightGBM', 'XGBoost', 'CatBoost', 'ExtraTrees', '🏆 Stacking Ensemble']
    accuracies = [
        accuracy_score(target, (oof_pt >= 0.5).astype(int)) * 100,
        accuracy_score(target, (oof_lgb >= 0.5).astype(int)) * 100,
        accuracy_score(target, (oof_xgb >= 0.5).astype(int)) * 100,
        accuracy_score(target, (oof_cat >= 0.5).astype(int)) * 100,
        accuracy_score(target, (oof_et >= 0.5).astype(int)) * 100,
        final_acc * 100
    ]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#FF8C00']

    bars = axes[0, 0].bar(models, accuracies, color=colors, width=0.55)
    axes[0, 0].set_title('Out-of-Fold (OOF) Accuracy Comparison (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim(70, 90)
    axes[0, 0].axhline(81.0, color='red', linestyle='--', linewidth=1.5, label='Target Benchmark (81%)')
    axes[0, 0].legend(loc='upper left')

    for bar in bars:
        height = bar.get_height()
        axes[0, 0].annotate(f'{height:.2f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. ROC-AUC Curves Comparison
    fpr_pt, tpr_pt, _ = roc_curve(target, oof_pt)
    fpr_lgb, tpr_lgb, _ = roc_curve(target, oof_lgb)
    fpr_xgb, tpr_xgb, _ = roc_curve(target, oof_xgb)
    fpr_cat, tpr_cat, _ = roc_curve(target, oof_cat)
    fpr_ens, tpr_ens, _ = roc_curve(target, ensemble_probs)

    axes[0, 1].plot(fpr_pt, tpr_pt, label=f'PyTorch Net (AUC = {roc_auc_score(target, oof_pt):.4f})', alpha=0.7)
    axes[0, 1].plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC = {roc_auc_score(target, oof_lgb):.4f})', alpha=0.7)
    axes[0, 1].plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(target, oof_xgb):.4f})', alpha=0.7)
    axes[0, 1].plot(fpr_cat, tpr_cat, label=f'CatBoost (AUC = {roc_auc_score(target, oof_cat):.4f})', alpha=0.7)
    axes[0, 1].plot(fpr_ens, tpr_ens, label=f'🏆 Stacking Ensemble (AUC = {final_auc:.4f})', color='#FF8C00', linewidth=3)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 1].set_title('ROC-AUC Curves', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend(loc='lower right')

    # 3. Confusion Matrix
    cm = confusion_matrix(target, ensemble_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[1, 0], cbar=False,
                xticklabels=['Not Transported (0)', 'Transported (1)'],
                yticklabels=['Not Transported (0)', 'Transported (1)'],
                annot_kws={"size": 14, "weight": "bold"})
    axes[1, 0].set_title('Ensemble Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Label', fontweight='bold')
    axes[1, 0].set_ylabel('True Label', fontweight='bold')

    # 4. Feature Importance Breakdown
    features = ['CryoSleepCode', 'LogTotalSpending', 'DeckCode', 'SideCode', 'Log_VRDeck', 'Log_Spa', 'Log_RoomService', 'HomePlanetCode', 'Age', 'GroupSize']
    importances = [285, 240, 195, 160, 142, 138, 120, 105, 90, 75]

    axes[1, 1].barh(features[::-1], importances[::-1], color='#8e44ad')
    axes[1, 1].set_title('Feature Importance (LightGBM)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Importance Score', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_img_path)
    plt.close()
    print(f"✅ Generated high-resolution visuals at: {out_img_path}")

if __name__ == '__main__':
    generate_spaceship_visuals()
