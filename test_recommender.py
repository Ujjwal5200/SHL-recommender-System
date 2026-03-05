import pandas as pd
import sys
sys.path.insert(0, '.')
from src.recommender_optimized import get_recommendations

# Load test data
test_df = pd.read_csv('data/test_predictions.csv')
unique_queries = test_df['query'].unique()

print('Testing recommender on unique queries...')
print('='*60)

for query in unique_queries[:3]:  # Test first 3 queries
    print(f'\nQuery: {query[:80]}...')
    print('-'*60)
    results = get_recommendations(query, top_n=5)
    print(f'Results ({len(results)}):')
    for i, r in enumerate(results):
        name = r.get('name', '')[:50]
        url = r.get('url', '')[:60]
        print(f'  {i+1}. {name}')
        print(f'     URL: {url}...')
    print()
