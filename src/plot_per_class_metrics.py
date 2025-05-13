import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def plot_per_class_metrics(pred_csv='results/evaluation/predictions.csv', out_dir='results/evaluation'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(pred_csv)
    y_true, y_pred = df['true_label'], df['predicted_label']
    report = classification_report(y_true, y_pred, output_dict=True)
    classes = list(report.keys())[:-3]
    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        plt.figure(figsize=(6,4))
        vals = [report[c][metric] for c in classes]
        ax = sns.barplot(x=classes, y=vals)
        plt.title(f'Per-Class {metric.capitalize()}')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Class')
        plt.ylim(0,1)
        # Annotate bars with percentage values INSIDE the bar
        for i, v in enumerate(vals):
            ax.text(i, v - 0.05, f"{v*100:.1f}%", ha='center', va='center', fontsize=11, fontweight='bold', color='white' if v > 0.5 else 'black')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'per_class_{metric}.png'))
        plt.close()
    # Per-class accuracy bar plot
    accs = []
    for c in classes:
        idx = (y_true == c)
        acc = np.mean(np.array(y_pred)[idx] == c)
        accs.append(acc)
    plt.figure(figsize=(6,4))
    ax = sns.barplot(x=classes, y=accs)
    plt.title('Per-Class Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.ylim(0,1)
    # Annotate bars with percentage values INSIDE the bar
    for i, v in enumerate(accs):
        ax.text(i, v - 0.05, f"{v*100:.1f}%", ha='center', va='center', fontsize=11, fontweight='bold', color='white' if v > 0.5 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_class_accuracy.png'))
    plt.close()
    # Error analysis: most common misclassifications
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    err = pd.DataFrame(cm, index=classes, columns=classes)
    err = err.copy()
    for c in classes:
        err.loc[c, c] = 0
    err_long = err.stack().reset_index()
    err_long.columns = ['True', 'Pred', 'Count']
    err_long = err_long[err_long['Count'] > 0].sort_values('Count', ascending=False)
    err_long.to_csv(os.path.join(out_dir, 'common_misclassifications.csv'), index=False)

if __name__ == "__main__":
    plot_per_class_metrics()
