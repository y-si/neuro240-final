import os
import pandas as pd
import argparse

def create_dataset_from_raw(raw_data_dir="data/raw", output_file="data/test_samples.csv"):
    """Create a CSV dataset from raw text files in specified directory"""
    data = []
    
    # Check if the directory exists
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"Directory not found: {raw_data_dir}")
    
    # List all author directories
    author_dirs = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
    
    if not author_dirs:
        raise ValueError(f"No author directories found in {raw_data_dir}")
    
    print(f"Found author directories: {', '.join(author_dirs)}")
    
    # Process each author directory
    for author in author_dirs:
        author_dir = os.path.join(raw_data_dir, author)
        files = [f for f in os.listdir(author_dir) if f.endswith('.txt')]
        
        print(f"Processing {len(files)} files for author: {author}")
        
        # Process each text file
        for filename in files:
            file_path = os.path.join(author_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    
                    # Skip empty files
                    if not text:
                        print(f"Skipping empty file: {file_path}")
                        continue
                    
                    data.append({
                        'text': text,
                        'label': author,
                        'filename': filename
                    })
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Created dataset with {len(df)} samples from {len(author_dirs)} authors")
    print(f"Saved to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Create a CSV dataset from raw text files")
    parser.add_argument("--input", type=str, default="data/raw", 
                        help="Directory containing author subdirectories with text files")
    parser.add_argument("--output", type=str, default="data/test_samples.csv",
                        help="Output CSV file path")
    
    args = parser.parse_args()
    create_dataset_from_raw(args.input, args.output)

if __name__ == "__main__":
    main() 