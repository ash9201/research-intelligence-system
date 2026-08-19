#!/usr/bin/env python3
"""
Check cross-encoder score output format and range
"""
from sentence_transformers import CrossEncoder
import numpy as np

# Load cross-encoder model
print("Loading cross-encoder model...")
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Test with sample pairs
pairs = [
    ["What is machine learning?", "Machine learning is a subset of AI"],
    ["What is machine learning?", "The weather is nice today"],
]

scores = model.predict(pairs)
print(f"Scores: {scores}")
print(f"Score type: {type(scores[0])}")
print(f"Score range: [{min(scores)}, {max(scores)}]")
print(f"Score dtype: {scores.dtype if hasattr(scores, 'dtype') else 'N/A'}")

# Cross-encoder outputs logits, not probabilities
# Need sigmoid transformation
sigmoid_scores = 1 / (1 + np.exp(-scores))
print(f"\nAfter sigmoid normalization: {sigmoid_scores}")
print(f"Sigmoid range: [{min(sigmoid_scores)}, {max(sigmoid_scores)}]")
