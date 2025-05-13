import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
csv_path = 'results/combined/model_comparison.csv'
df = pd.read_csv(csv_path)

plt.figure(figsize=(10,6))
ax = sns.barplot(x='model', y='mean_accuracy', data=df, yerr=df['std_accuracy'], palette='deep')
plt.title('Model Mean Accuracy with Std Dev')
plt.ylabel('Mean Accuracy')
plt.ylim(0.4,1.0)
plt.xlabel('Model')
plt.xticks(rotation=30, ha='right')
for i, v in enumerate(df['mean_accuracy']):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('results/combined/model_accuracy_pub.png')
print('Saved to results/combined/model_accuracy_pub.png')
