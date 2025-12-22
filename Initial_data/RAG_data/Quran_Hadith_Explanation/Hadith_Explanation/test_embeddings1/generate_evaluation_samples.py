"""
Generate Evaluation Samples for Hadith Shar7 Embeddings
========================================================
Creates a structured JSON file with queries and their top-5 retrieved chunks
for evaluation and sharing with colleagues.
"""

import pickle
import json
import numpy as np
import re
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
print(SCRIPT_DIR)
CHUNKS_PKL = SCRIPT_DIR /"shar7_chunks_complete.pkl"
OUTPUT_JSON = SCRIPT_DIR /"evaluation_samples.json"

# Number of chunks to retrieve per query
TOP_K = 5

# ============================================================================
# TEST QUERIES
# ============================================================================

# Original 5 queries from test_embeddings1.py
ORIGINAL_QUERIES = [
    "ما حكم لبس الحرير للرجال",  # Ruling on silk for men
    "آداب العيد في الإسلام",  # Eid etiquette in Islam
    "الأكل قبل صلاة العيد",  # Eating before Eid prayer
    "معاملة الزوجة بالرفق",  # Treating wife with kindness
    "اللهو المباح في الإسلام",  # Permissible entertainment in Islam
]

# Challenging queries - edge cases and complex questions
CHALLENGING_QUERIES = [
    # Semantic understanding tests
    "هل يجوز الغناء والموسيقى",  # Is singing/music permissible
    "حكم الذهب للرجال",  # Gold ruling for men (similar to silk)
    "كيف نربي الأبناء",  # How to raise children
    
    # Specific fiqh questions
    "ما هي سنن العيد",  # What are the Sunnahs of Eid
    "حكم صيام يوم العيد",  # Ruling on fasting on Eid day
    
    # Abstract/conceptual queries
    "التوازن بين الدين والدنيا",  # Balance between religion and worldly life
    "الرحمة في الإسلام",  # Mercy in Islam
    
    # Queries with different wording (paraphrase test)
    "ارتداء الملابس الفاخرة",  # Wearing luxurious clothes (related to silk hadith)
    "الفرح والسرور في المناسبات",  # Joy and happiness in occasions
    
    
]

ALL_QUERIES = ORIGINAL_QUERIES + CHALLENGING_QUERIES

# ============================================================================
# ARABIC PREPROCESSING
# ============================================================================

class ArabicPreprocessor:
    def __init__(self):
        self.arabic_normalizations = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
            'ة': 'ه', 'ى': 'ي', 'ؤ': 'و', 'ئ': 'ي',
        }
        self.tashkeel_pattern = re.compile(r'[\u064B-\u065F\u0670]')
    
    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace('ـ', '')
        text = self.tashkeel_pattern.sub('', text)
        for orig, norm in self.arabic_normalizations.items():
            text = text.replace(orig, norm)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("GENERATING EVALUATION SAMPLES")
    print("="*70)
    
    # Check if embeddings exist
    if not CHUNKS_PKL.exists():
        print(f"\n❌ ERROR: Embeddings file not found!")
        print(f"   Expected: {CHUNKS_PKL}")
        print(f"   Please run test_embeddings1.py first.")
        return
    
    # Load embeddings
    print(f"\n📂 Loading embeddings from: {CHUNKS_PKL}")
    with open(CHUNKS_PKL, 'rb') as f:
        data = pickle.load(f)
    
    chunks = data['chunks']
    embeddings = data['embeddings']
    model_name = data['model_name']
    
    print(f"✓ Loaded {len(chunks)} chunks")
    print(f"✓ Model: {model_name}")
    
    # Load model
    print(f"\n🔄 Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    print(f"✓ Model loaded")
    
    # Check if E5 model
    is_e5 = 'e5' in model_name.lower()
    
    # Preprocessor
    preprocessor = ArabicPreprocessor()
    
    # Generate results
    print(f"\n🔍 Processing {len(ALL_QUERIES)} queries...")
    
    evaluation_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": model_name,
            "total_chunks": len(chunks),
            "top_k": TOP_K,
            "num_original_queries": len(ORIGINAL_QUERIES),
            "num_challenging_queries": len(CHALLENGING_QUERIES),
        },
        "queries": []
    }
    
    for i, query in enumerate(ALL_QUERIES):
        query_type = "original" if i < len(ORIGINAL_QUERIES) else "challenging"
        
        print(f"\n   [{i+1}/{len(ALL_QUERIES)}] {query}")
        
        # Preprocess query
        processed_query = preprocessor.preprocess(query)
        
        # Add prefix for E5 models
        if is_e5:
            query_for_embedding = f"query: {processed_query}"
        else:
            query_for_embedding = processed_query
        
        # Get query embedding
        query_embedding = model.encode(
            query_for_embedding,
            normalize_embeddings=True
        )
        
        # Calculate similarities
        similarities = np.dot(embeddings, query_embedding)
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:TOP_K]
        
        # Build results
        retrieved_chunks = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = chunks[idx]
            score = float(similarities[idx])
            
            retrieved_chunks.append({
                "rank": rank,
                "similarity_score": round(score, 4),
                "sharhId": chunk['sharhId'],
                "hadith_text": chunk['hadith_text'],
                "shar7_text": chunk['shar7_text'],
                "metadata": chunk['metadata']
            })
        
        query_result = {
            "query": query,
            "query_type": query_type,
            "query_preprocessed": processed_query,
            "retrieved_chunks": retrieved_chunks
        }
        
        evaluation_data["queries"].append(query_result)
        
        # Print top result score
        print(f"      Top score: {retrieved_chunks[0]['similarity_score']:.4f}")
    
    # Save to JSON
    print(f"\n💾 Saving to: {OUTPUT_JSON}")
    
    # Ensure output directory exists
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved evaluation samples")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n📊 Generated evaluation for {len(ALL_QUERIES)} queries:")
    print(f"   - {len(ORIGINAL_QUERIES)} original queries")
    print(f"   - {len(CHALLENGING_QUERIES)} challenging queries")
    print(f"   - {TOP_K} chunks retrieved per query")
    print(f"\n📁 Output: {OUTPUT_JSON}")
    print(f"\nShare this file with your friend for evaluation!")

if __name__ == "__main__":
    main()
