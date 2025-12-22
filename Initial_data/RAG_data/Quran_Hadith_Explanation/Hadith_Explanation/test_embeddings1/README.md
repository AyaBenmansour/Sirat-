# Hadith Shar7 Embedding Pipeline

## Overview
This pipeline embeds hadith explanations (shar7) for semantic search using state-of-the-art multilingual models.

## Model Selection Rationale

### Recommended: `intfloat/multilingual-e5-large`
- **Why**: Best performance on Arabic semantic similarity tasks
- **Dimensions**: 1024 (rich representations)
- **Training**: Contrastive learning on 100+ languages including Arabic
- **Special**: Requires "query: " / "passage: " prefixes for optimal results

### Alternatives:
| Model | Dims | Arabic Quality | Speed | Notes |
|-------|------|----------------|-------|-------|
| multilingual-e5-large | 1024 | ⭐⭐⭐⭐⭐ | Medium | Best for Islamic texts |
| multilingual-e5-base | 768 | ⭐⭐⭐⭐ | Fast | Good balance |
| paraphrase-multilingual-mpnet | 768 | ⭐⭐⭐ | Fast | General purpose |
| aubmindlab/bert-base-arabertv2 | 768 | ⭐⭐⭐⭐ | Fast | Arabic-only, needs fine-tuning |

## Arabic Preprocessing Pipeline

### Why Preprocessing Matters for Arabic:
1. **Tashkeel (diacritics)**: Same word can be written with/without harakat
2. **Character variations**: أ/إ/آ/ا are often used interchangeably
3. **Tatweel**: ـ is used for visual elongation but adds noise

### Our Pipeline:
```
Input → Remove URLs → Remove Tatweel → Remove Tashkeel → Normalize Chars → Normalize Whitespace → Output
```

### What We Preserve:
- Islamic honorifics (صلى الله عليه وسلم)
- Scholarly terminology
- Sentence structure

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python test_embeddings1.py
```

## Output Files

1. **shar7.json**: Extracted chunks with metadata
2. **shar7_chunks_complete.pkl**: Complete data with embeddings

## Loading Embeddings for Search

```python
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load saved data
with open('shar7_chunks_complete.pkl', 'rb') as f:
    data = pickle.load(f)

chunks = data['chunks']
embeddings = data['embeddings']
model = SentenceTransformer(data['model_name'])

# Search function
def search(query, top_k=5):
    # For E5 models, add query prefix
    query_emb = model.encode(f"query: {query}", normalize_embeddings=True)
    scores = np.dot(embeddings, query_emb)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], scores[i]) for i in top_idx]

# Example
results = search("حكم الصيام في السفر")
for chunk, score in results:
    print(f"Score: {score:.4f}")
    print(f"Hadith: {chunk['hadith_text'][:100]}...")
```

## Tips & Insights

### For Better Results:
1. **Chunk size**: Our shar7 texts are already good chunks (avg ~1000 chars)
2. **Query preprocessing**: Apply same preprocessing to queries
3. **Hybrid search**: Combine with BM25 for keyword matching

### GPU Acceleration:
```python
# Force GPU usage
model = SentenceTransformer(model_name, device='cuda')
```

### Memory Optimization:
```python
# For large datasets, use float16
embeddings = embeddings.astype(np.float16)
```
