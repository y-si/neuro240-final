import pandas as pd
from sklearn.model_selection import train_test_split
import hashlib

# Load your main dataset (adjust path if needed)
df = pd.read_csv('data/test_samples.csv')

# Use the same split logic as your training pipeline
y = df['author'] if 'author' in df.columns else df['label']
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=y)

# Hash function for text
hash_text = lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest()

train_hashes = set(train_df['text'].map(hash_text))
test_hashes = set(test_df['text'].map(hash_text))

# Find overlaps
leak_hashes = train_hashes & test_hashes

print(f"Train set size: {len(train_df)} | Test set size: {len(test_df)}")
print(f"Number of overlapping texts between train and test: {len(leak_hashes)}")

if leak_hashes:
    print("Example leaked texts:")
    leaked_texts = test_df[test_df['text'].map(hash_text).isin(leak_hashes)]['text'].unique()
    for t in leaked_texts[:5]:
        print(f"---\n{t}\n---")
else:
    print("No duplicate texts between train and test.")
