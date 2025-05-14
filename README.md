# Authorship Attribution Model Training

This repository contains code for training and evaluating authorship attribution models using stylometric features and transformer-based embeddings.

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Place your text data in the `data/raw` directory with the following structure:
```
data/raw/
  ├── gpt4/
  │   ├── sample1.txt
  │   ├── sample2.txt
  │   └── ...
  ├── claude/
  │   ├── sample1.txt
  │   ├── sample2.txt
  │   └── ...
  └── llama/
      ├── sample1.txt
      ├── sample2.txt
      └── ...
```

Each directory name represents a model/author label, and each text file contains a text sample.

## Model Training

To train models with different classifiers (Logistic Regression, Random Forest, Neural Network):

```bash
python src/train_models.py
```

This will:
1. Load the data from `data/raw`
2. Extract stylometric features and transformer-based embeddings
3. Train multiple classification models
4. Save trained models to the `models` directory
5. Generate evaluation results in the `results` directory

## Model Evaluation

To evaluate a trained model on new data:

```bash
python src/evaluate_model.py --model models/stylometric/random_forest.pkl --data path/to/new_data.csv --output results/evaluation
```

Arguments:
- `--model`: Path to the trained model file (.pkl)
- `--data`: Path to the CSV file containing new data (must have 'text' and 'label' columns)
- `--output`: Directory to save evaluation results (default: results/evaluation)

## Features Used

The system extracts the following types of features:

1. **Stylometric Features**:
   - Text length metrics (word count, character count)
   - Punctuation usage
   - Sentence structure (length, count, variation)
   - Lexical diversity
   - Capitalization patterns

2. **Transformer Embeddings**:
   - RoBERTa embeddings
   - XLNet embeddings

## Results

After training, you can find:
- Trained models in the `models` directory
- Confusion matrices in the `results` directory
- Feature importance plots
- Classification reports
- Model comparison CSV

## Documentation

For detailed technical explanations, advanced usage, and step-by-step workflows, please see the [full project documentation](./documentation.md). This file contains in-depth guides and script-by-script instructions beyond the quickstart and overview provided here.

## Reference

This implementation is inspired by the work of Uchendu et al. in their paper "Authorship Attribution for Neural Text Generation" and their repository:
https://github.com/AdaUchendu/Authorship-Attribution-for-Neural-Text-Generation 

I also acknowledge the use of ChatGPT and Windsurf throughout this project. 