"""
Semantic Search Utility for Hadith Shar7
=========================================
Use this script to search through embedded shar7 after running test_embeddings1.py
"""

import pickle
import numpy as np
import re
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CHUNKS_PKL = Path(__file__).parent / "shar7_chunks_complete.pkl"

# ============================================================================
# ARABIC PREPROCESSING (same as main script)
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
# SEARCH CLASS
# ============================================================================

class Shar7Searcher:
    def __init__(self, pkl_path: str = None):
        """Initialize searcher with saved embeddings."""
        pkl_path = pkl_path or CHUNKS_PKL
        
        print(f"Loading embeddings from: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.embeddings = data['embeddings']
        self.model_name = data['model_name']
        
        print(f"Loaded {len(self.chunks)} chunks")
        print(f"Model: {self.model_name}")
        
        # Load model
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)
        self.preprocessor = ArabicPreprocessor()
        
        # Check if E5 model
        self.is_e5 = 'e5' in self.model_name.lower()
    
    def search(self, query: str, top_k: int = 5) -> list:
        """
        Search for relevant shar7 given a query.
        
        Args:
            query: Arabic search query
            top_k: Number of results to return
        
        Returns:
            List of (chunk, score) tuples
        """
        # Preprocess query
        processed_query = self.preprocessor.preprocess(query)
        
        # Add prefix for E5 models
        if self.is_e5:
            query_for_embedding = f"query: {processed_query}"
        else:
            query_for_embedding = processed_query
        
        # Get query embedding
        query_embedding = self.model.encode(
            query_for_embedding,
            normalize_embeddings=True
        )
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.chunks[i], similarities[i]) for i in top_indices]
    
    def print_results(self, results: list):
        """Pretty print search results."""
        for rank, (chunk, score) in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"Result #{rank} | Score: {score:.4f}")
            print(f"{'='*60}")
            print(f"📖 Book: {chunk['metadata']['book']}")
            print(f"👤 Rawi: {chunk['metadata']['rawi']}")
            print(f"📝 Grade: {chunk['metadata']['grade'][:50]}..." if chunk['metadata']['grade'] else "")
            print(f"\n🔹 Hadith:")
            print(f"   {chunk['hadith_text'][:200]}...")
            print(f"\n🔸 Shar7 (explanation):")
            print(f"   {chunk['shar7_text'][:300]}...")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_search():
    """Run interactive search session."""
    print("\n" + "="*60)
    print("HADITH SHAR7 SEMANTIC SEARCH")
    print("="*60)
    
    searcher = Shar7Searcher()
    
    print("\n✓ Ready for search!")
    print("Type your query in Arabic (or 'quit' to exit)\n")
    
    while True:
        query = input("\n🔍 Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        results = searcher.search(query, top_k=3)
        searcher.print_results(results)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    interactive_search()
