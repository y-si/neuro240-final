import pandas as pd
import glob
import os

# Directory containing misclassified CSVs
MISCLASS_DIR = os.path.join(os.path.dirname(__file__), '../results/combined')

files = glob.glob(os.path.join(MISCLASS_DIR, '*_misclassified.csv'))
summary = []
for f in files:
    model = os.path.basename(f).replace('_misclassified.csv', '')
    try:
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            summary.append({
                'model': model,
                'true_label': row['true_label'],
                'predicted_label': row['predicted_label'],
                'text_snippet': str(row['text'])[:120].replace('\n', ' ')
            })
    except Exception as e:
        print(f'Error reading {f}: {e}')

summary_df = pd.DataFrame(summary)
summary_csv = os.path.join(MISCLASS_DIR, 'misclass_summary.csv')
summary_df.to_csv(summary_csv, index=False)
print(f'Saved summary to {summary_csv}')

# Show top patterns
if not summary_df.empty:
    print('Top 10 misclassifications:')
    print(summary_df.groupby(['true_label','predicted_label']).size().sort_values(ascending=False).head(10))
    print('\nSample misclassified texts:')
    print(summary_df.head(10))
else:
    print('No misclassifications found.')
