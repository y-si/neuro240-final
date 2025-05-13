import pandas as pd
import re
import numpy as np

TECH_TERMS = [
    'algorithm', 'data structure', 'complexity', 'runtime', 'compile', 'recursion', 'stack', 'queue', 'tree', 'graph',
    'O(', 'DFS', 'BFS', 'node', 'edge', 'vertex', 'search', 'traverse', 'sort', 'merge', 'split', 'class', 'object',
    'inheritance', 'encapsulation', 'polymorphism', 'interface', 'abstraction', 'function', 'method', 'variable',
    'parameter', 'argument', 'type', 'static', 'dynamic', 'compile-time', 'run-time', 'exception', 'error', 'bug',
    'debug', 'test', 'assert', 'unit test', 'integration', 'dependency', 'library', 'module', 'package', 'import',
    'export', 'API', 'SDK', 'framework', 'design pattern', 'SOLID', 'principle', 'architecture', 'component', 'system'
]

def technicality_score(text):
    text = text.lower()
    count = sum(1 for term in TECH_TERMS if term.lower() in text)
    return count / max(1, len(text.split()))

def add_technicality_feature(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['technicality_score'] = df['text'].apply(technicality_score)
    df.to_csv(output_csv, index=False)
    print(f'Saved with technicality_score to {output_csv}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: python feature_engineering_technicality.py input.csv output.csv')
    else:
        add_technicality_feature(sys.argv[1], sys.argv[2])
