import numpy as np
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer, AutoModel, XLNetTokenizer, XLNetModel
import torch
import nltk
from nltk.tokenize import sent_tokenize
import statistics
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)  # This is the correct package for sentence tokenization
nltk.download('averaged_perceptron_tagger', quiet=True)

def extract_stylometric_features(df):
    # Basic features
    df["sentence_length"] = df["text"].apply(lambda x: len(x.split()))
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["char_count"] = df["text"].apply(len)
    df["avg_word_length"] = df["text"].apply(lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)
    
    # Punctuation features - Use manual counting to avoid regex issues
    df["period_count"] = df["text"].apply(lambda x: x.count('.'))
    df["exclamation_count"] = df["text"].apply(lambda x: x.count('!'))
    df["question_count"] = df["text"].apply(lambda x: x.count('?'))
    df["punctuation_count"] = df["period_count"] + df["exclamation_count"] + df["question_count"]
    df["comma_count"] = df["text"].apply(lambda x: x.count(','))
    df["semicolon_count"] = df["text"].apply(lambda x: x.count(';'))
    df["colon_count"] = df["text"].apply(lambda x: x.count(':'))
    df["double_quote_count"] = df["text"].apply(lambda x: x.count('"'))
    df["single_quote_count"] = df["text"].apply(lambda x: x.count("'"))
    df["quote_count"] = df["double_quote_count"] + df["single_quote_count"]
    
    # Digit and symbol counts
    df["digit_count"] = df["text"].apply(lambda x: sum(c.isdigit() for c in x))
    df["symbol_count"] = df["text"].apply(lambda x: sum(not c.isalnum() and not c.isspace() for c in x))
    
    # Longest word length
    df["longest_word"] = df["text"].apply(lambda x: max([len(word) for word in x.split()], default=0))
    
    # Stopword ratio
    df["stopword_ratio"] = df["text"].apply(lambda x: sum(word.lower() in ENGLISH_STOP_WORDS for word in x.split()) / (len(x.split()) if len(x.split()) > 0 else 1))
    
    # Function word frequencies (select 10 common ones)
    function_words = ["the", "and", "of", "to", "in", "that", "is", "for", "on", "with"]
    for fw in function_words:
        df[f"fw_{fw}"] = df["text"].apply(lambda x: x.lower().split().count(fw))
    
    # Sentence-level features using simple split on punctuation if NLTK fails
    try:
        df["sentences"] = df["text"].apply(sent_tokenize)
    except LookupError:
        print("NLTK punkt not available, using simple sentence splitting")
        def simple_sent_tokenize(text):
            sentences = re.split(r'[.!?]\s+', text)
            return [s for s in sentences if s.strip()]
        df["sentences"] = df["text"].apply(simple_sent_tokenize)
    df["sentence_count"] = df["sentences"].apply(len)
    def safe_mean(values):
        return np.mean(values) if values else 0
    def safe_stdev(values):
        return statistics.stdev(values) if len(values) > 1 else 0
    df["avg_sentence_length"] = df["sentences"].apply(
        lambda sentences: safe_mean([len(s.split()) for s in sentences])
    )
    df["std_sentence_length"] = df["sentences"].apply(
        lambda sentences: safe_stdev([len(s.split()) for s in sentences])
    )
    # Lexical diversity
    df["unique_words"] = df["text"].apply(lambda x: len(set(x.lower().split())))
    df["lexical_diversity"] = df.apply(
        lambda row: row["unique_words"] / row["word_count"] if row["word_count"] > 0 else 0,
        axis=1
    )
    # Uppercase features
    df["uppercase_count"] = df["text"].apply(lambda x: sum(1 for c in x if c.isupper()))
    df["uppercase_ratio"] = df.apply(
        lambda row: row["uppercase_count"] / row["char_count"] if row["char_count"] > 0 else 0,
        axis=1
    )
    # POS tag counts
    try:
        def pos_counts(text):
            tokens = nltk.word_tokenize(text)
            tags = nltk.pos_tag(tokens)
            counts = {"NN": 0, "VB": 0, "JJ": 0, "RB": 0}
            for word, tag in tags:
                if tag.startswith("NN"): counts["NN"] += 1
                if tag.startswith("VB"): counts["VB"] += 1
                if tag.startswith("JJ"): counts["JJ"] += 1
                if tag.startswith("RB"): counts["RB"] += 1
            return counts["NN"], counts["VB"], counts["JJ"], counts["RB"]
        df[["noun_count", "verb_count", "adj_count", "adv_count"]] = df["text"].apply(
            lambda x: pd.Series(pos_counts(x))
        )
    except Exception:
        df["noun_count"] = 0
        df["verb_count"] = 0
        df["adj_count"] = 0
        df["adv_count"] = 0
    # Readability metrics
    try:
        import textstat
        df["flesch_reading_ease"] = df["text"].apply(lambda x: textstat.flesch_reading_ease(x))
        df["gunning_fog"] = df["text"].apply(lambda x: textstat.gunning_fog(x))
    except ImportError:
        df["flesch_reading_ease"] = 0
        df["gunning_fog"] = 0
    # Advanced stylometric features
    from collections import Counter
    def char_ngrams(text, n):
        text = text.replace(' ', '')
        return [text[i:i+n] for i in range(len(text)-n+1)]
    # Top 5 most common bigrams and trigrams
    for n in [2, 3]:
        ngram_counts = Counter()
        for text in df["text"]:
            ngram_counts.update(char_ngrams(text.lower(), n))
        most_common = [ng for ng, _ in ngram_counts.most_common(5)]
        for ng in most_common:
            df[f"char_{n}gram_{ng}"] = df["text"].apply(lambda x: char_ngrams(x.lower(), n).count(ng))
    # Word bigram/trigram frequencies
    def word_ngrams(text, n):
        words = text.lower().split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    for n in [2, 3]:
        ngram_counts = Counter()
        for text in df["text"]:
            ngram_counts.update(word_ngrams(text, n))
        most_common = [ng for ng, _ in ngram_counts.most_common(5)]
        for ng in most_common:
            df[f"word_{n}gram_{ng.replace(' ', '_')}"] = df["text"].apply(lambda x: word_ngrams(x, n).count(ng))
    # Punctuation ratio
    df["punctuation_ratio"] = df.apply(lambda row: row["punctuation_count"] / row["char_count"] if row["char_count"] > 0 else 0, axis=1)
    # Average syllables per word
    try:
        import textstat
        df["avg_syllables_per_word"] = df["text"].apply(lambda x: textstat.syllable_count(x) / (len(x.split()) if len(x.split()) > 0 else 1))
    except ImportError:
        df["avg_syllables_per_word"] = 0
    # Hapax legomena ratio
    def hapax_ratio(text):
        words = text.lower().split()
        counts = Counter(words)
        hapax = sum(1 for v in counts.values() if v == 1)
        return hapax / (len(words) if len(words) > 0 else 1)
    df["hapax_legomena_ratio"] = df["text"].apply(hapax_ratio)
    # Yule's K (lexical richness)
    def yules_k(text):
        words = text.lower().split()
        N = len(words)
        if N == 0:
            return 0
        freq = Counter(words)
        M1 = N
        M2 = sum([v*v for v in freq.values()])
        return 10000 * (M2 - M1) / (M1*M1)
    df["yules_k"] = df["text"].apply(yules_k)
    # Entropy of character distribution
    import math
    def char_entropy(text):
        text = text.replace(' ', '').lower()
        if not text:
            return 0
        freq = Counter(text)
        total = len(text)
        return -sum((count/total) * math.log2(count/total) for count in freq.values())
    df["char_entropy"] = df["text"].apply(char_entropy)
    
    # --- Add Sentence Embeddings (Sentence Transformers) ---
    def extract_sentence_embeddings(texts, model_name="all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_name)
        embeddings = model.encode(list(texts), show_progress_bar=True)
        return embeddings

    # Add sentence embeddings as features
    try:
        sent_embs = extract_sentence_embeddings(df["text"])
        sent_emb_cols = [f"sent_emb_{i}" for i in range(sent_embs.shape[1])]
        sent_emb_df = pd.DataFrame(sent_embs, columns=sent_emb_cols)
        df = pd.concat([df.reset_index(drop=True), sent_emb_df.reset_index(drop=True)], axis=1)
    except Exception as e:
        print(f"Could not extract sentence embeddings: {e}")
    
    # Drop intermediate columns
    df = df.drop(columns=["sentences", "period_count", "double_quote_count", "single_quote_count"], errors="ignore")
    # Select all feature columns (exclude text and label)
    feature_columns = [col for col in df.columns if col not in ["text", "label"]]
    return df[feature_columns]

def extract_roberta_embeddings(texts, pooling='cls'):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base")

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        if pooling == 'cls':
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        elif pooling == 'mean':
            # Use mean of token embeddings
            attention_mask = inputs['attention_mask']
            embedding = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
            embedding = embedding.squeeze().numpy()
        
        embeddings.append(embedding)

    return np.array(embeddings)

def extract_xlnet_embeddings(texts, pooling='last'):
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    model = XLNetModel.from_pretrained("xlnet-base-cased")

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        if pooling == 'last':
            # Use last token embedding (XLNet processes right-to-left)
            embedding = outputs.last_hidden_state[:, -1, :].squeeze().numpy()
        elif pooling == 'mean':
            # Use mean of token embeddings
            attention_mask = inputs['attention_mask']
            embedding = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
            embedding = embedding.squeeze().numpy()
        
        embeddings.append(embedding)

    return np.array(embeddings)

def extract_tfidf_features(df, max_features=500):
    """Extract TF-IDF features from the text column of the DataFrame."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
    return tfidf_df, vectorizer

def combine_features(df, model_embeddings=None, tfidf_vectorizer=None):
    """
    Combine stylometric features with model embeddings and TF-IDF features
    
    Args:
        df: DataFrame with text and features
        model_embeddings: Optional dictionary mapping model names to embeddings arrays
        tfidf_vectorizer: Optional fitted TfidfVectorizer. If None, will fit a new one.
    
    Returns:
        DataFrame with combined features, fitted TfidfVectorizer
    """
    # Get stylometric features
    features_df = extract_stylometric_features(df)
    
    # Add TF-IDF features
    if tfidf_vectorizer is None:
        tfidf_df, tfidf_vectorizer = extract_tfidf_features(df)
    else:
        tfidf_matrix = tfidf_vectorizer.transform(df['text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
    features_df = pd.concat([features_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    
    # If model embeddings are provided, add them as columns
    if model_embeddings:
        for model_name, embeddings in model_embeddings.items():
            # Convert embeddings to DataFrame
            embedding_df = pd.DataFrame(
                embeddings,
                columns=[f"{model_name}_emb_{i}" for i in range(embeddings.shape[1])]
            )
            # Concatenate with features DataFrame
            features_df = pd.concat([features_df.reset_index(drop=True), 
                                    embedding_df.reset_index(drop=True)], 
                                    axis=1)
    
    return features_df, tfidf_vectorizer
