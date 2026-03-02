# test_recommend.py
from src.recommender import get_recommendations

query = "I am hiring for Java developers who can also collaborate effectively with my business teams."

recs = get_recommendations(query)

print(f"Got {len(recs)} recommendations:")
for r in recs:
    print(f"- {r['name']}")
    print(f"  URL: {r['url']}")
    print(f"  Types: {r.get('test_types')}")