"""
Fatawa Embedding Pipeline
=========================
This script processes fatawa (Islamic Q&A) for semantic search using embeddings.
Following the same pipeline as hadith shar7 embeddings.

Best Embedding Models for Arabic/Islamic Content:
-------------------------------------------------
1. intfloat/multilingual-e5-large (RECOMMENDED)
   - 1024 dimensions, excellent Arabic support
   - Trained on diverse multilingual data
   - Best semantic understanding for religious texts

Arabic Preprocessing Pipeline:
------------------------------
1. Normalize Arabic characters (أإآ → ا, ة → ه, etc.)
2. Remove tashkeel (diacritics/harakat)
3. Remove tatweel (kashida elongation)
4. Normalize whitespace
5. Keep Islamic terms intact (صلى الله عليه وسلم, رضي الله عنه, etc.)
"""

import json
import pickle
import re
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output paths
INPUT_JSON = Path(__file__).parent / "fatawa_for_embeddings_cleaned.json"
OUTPUT_DIR = Path(__file__).parent
FATAWA_JSON = OUTPUT_DIR / "fatawa_processed.json"
CHUNKS_PKL = OUTPUT_DIR / "fatawa_chunks_complete.pkl"

# Embedding model - Best for Arabic/Islamic content
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Batch size for embedding (adjust based on GPU memory)
BATCH_SIZE = 32

# ============================================================================
# ARABIC TEXT PREPROCESSING
# ============================================================================

class ArabicPreprocessor:
    """
    Arabic text preprocessing pipeline optimized for Islamic scholarly texts.
    """
    
    def __init__(self):
        # Arabic character normalization mappings
        self.arabic_normalizations = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',  # Alef variations
            'ة': 'ه',  # Ta marbuta → Ha
            'ى': 'ي',  # Alef maksura → Ya
            'ؤ': 'و',  # Waw with hamza
            'ئ': 'ي',  # Ya with hamza
        }
        
        # Tashkeel (diacritics) pattern - all Arabic diacritical marks
        self.tashkeel_pattern = re.compile(r'[\u064B-\u065F\u0670]')
        
        # Tatweel (kashida) - elongation character
        self.tatweel = 'ـ'
    
    def remove_tashkeel(self, text: str) -> str:
        """Remove Arabic diacritical marks (harakat)."""
        return self.tashkeel_pattern.sub('', text)
    
    def remove_tatweel(self, text: str) -> str:
        """Remove tatweel/kashida elongation."""
        return text.replace(self.tatweel, '')
    
    def normalize_arabic_chars(self, text: str) -> str:
        """Normalize Arabic character variations."""
        for original, normalized in self.arabic_normalizations.items():
            text = text.replace(original, normalized)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and newlines."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_non_arabic_noise(self, text: str) -> str:
        """Remove URLs, emails, and excessive punctuation."""
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'([.!?،؛])\1+', r'\1', text)
        return text
    
    def preprocess(self, text: str, keep_tashkeel: bool = False) -> str:
        """Full preprocessing pipeline."""
        if not text:
            return ""
        
        text = self.remove_non_arabic_noise(text)
        text = self.remove_tatweel(text)
        
        if not keep_tashkeel:
            text = self.remove_tashkeel(text)
        
        text = self.normalize_arabic_chars(text)
        text = self.normalize_whitespace(text)
        
        return text

# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_fatawa_chunks(input_path: Path) -> list:
    """
    Extract fatawa (question + answer pairs) with metadata from JSON.
    
    Each chunk contains:
    - combined_text: Question + Answer combined for embedding
    - arabic_question: Original question
    - arabic_answer: Original answer
    - fatawa_id: Unique identifier
    """
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING FATAWA CHUNKS")
    print("="*70)
    
    print(f"\n📂 Loading data from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} fatawa entries")
    
    chunks = []
    skipped = 0
    
    for item in data:
        question = item.get('arabic_question', '')
        answer = item.get('arabic_answer', '')
        fatawa_id = item.get('fatawa_id', '')
        
        # Skip empty entries
        if not question or not answer or len(question.strip()) < 5 or len(answer.strip()) < 10:
            skipped += 1
            continue
        
        # Combine question and answer for embedding
        combined_text = f"السؤال: {question}\n\nالجواب: {answer}"
        
        chunk = {
            'fatawa_id': fatawa_id,
            'arabic_question': question,
            'arabic_answer': answer,
            'combined_text': combined_text,
        }
        chunks.append(chunk)
    
    print(f"✓ Extracted {len(chunks)} valid fatawa chunks")
    print(f"⚠ Skipped {skipped} entries (empty or too short)")
    
    # Statistics
    q_lengths = [len(c['arabic_question']) for c in chunks]
    a_lengths = [len(c['arabic_answer']) for c in chunks]
    combined_lengths = [len(c['combined_text']) for c in chunks]
    
    print(f"\n📊 Fatawa Statistics:")
    print(f"   Questions - Min: {min(q_lengths)}, Max: {max(q_lengths)}, Avg: {sum(q_lengths)//len(q_lengths)} chars")
    print(f"   Answers   - Min: {min(a_lengths)}, Max: {max(a_lengths)}, Avg: {sum(a_lengths)//len(a_lengths)} chars")
    print(f"   Combined  - Min: {min(combined_lengths)}, Max: {max(combined_lengths)}, Avg: {sum(combined_lengths)//len(combined_lengths)} chars")
    
    return chunks

# ============================================================================
# EMBEDDING
# ============================================================================

def load_embedding_model(model_name: str):
    """Load the sentence transformer model."""
    print("\n" + "="*70)
    print("STEP 3: LOADING EMBEDDING MODEL")
    print("="*70)
    
    print(f"\n🔄 Loading model: {model_name}")
    print("   (This may take a few minutes on first run...)")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n❌ Error: sentence-transformers not installed!")
        print("   Run: pip install sentence-transformers")
        sys.exit(1)
    
    model = SentenceTransformer(model_name)
    
    print(f"✓ Model loaded successfully")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"   Max sequence length: {model.max_seq_length}")
    
    return model

def create_embeddings(model, texts: list, batch_size: int = 32) -> np.ndarray:
    """
    Create embeddings for texts using the model.
    
    For E5 models, we need to add "query: " or "passage: " prefix.
    For passages (our fatawa texts), we use "passage: " prefix.
    """
    print("\n" + "="*70)
    print("STEP 4: CREATING EMBEDDINGS")
    print("="*70)
    
    print(f"\n🔄 Embedding {len(texts)} texts...")
    print(f"   Batch size: {batch_size}")
    
    # For E5 models, add passage prefix
    if 'e5' in EMBEDDING_MODEL.lower():
        print("   Adding 'passage: ' prefix for E5 model")
        texts = [f"passage: {t}" for t in texts]
    
    start_time = datetime.now()
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize for cosine similarity
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n✓ Embeddings created successfully")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Time: {elapsed:.2f} seconds")
    print(f"   Speed: {len(texts)/elapsed:.1f} texts/second")
    
    return embeddings

# ============================================================================
# EVALUATION / TESTING
# ============================================================================

def test_embeddings(model, chunks: list, embeddings: np.ndarray):
    """
    Test the embeddings with sample queries to verify semantic search quality.
    """
    print("\n" + "="*70)
    print("STEP 5: TESTING EMBEDDINGS")
    print("="*70)
    
    # Test queries - Islamic fatawa related
    test_queries = [
        "ما حكم الربا في الإسلام",  # Ruling on usury
        "كيفية صلاة الجماعة",  # How to pray in congregation
        "حكم الزكاة على المال",  # Ruling on zakat
        "ما هي شروط الحج",  # Conditions of Hajj
          # Ruling on circumcision
    ]
    
    print(f"\n🔍 Testing with {len(test_queries)} queries...\n")
    
    preprocessor = ArabicPreprocessor()
    
    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"📝 Query: {query}")
        
        processed_query = preprocessor.preprocess(query)
        
        # For E5 models, add query prefix
        if 'e5' in EMBEDDING_MODEL.lower():
            query_for_embedding = f"query: {processed_query}"
        else:
            query_for_embedding = processed_query
        
        query_embedding = model.encode(
            query_for_embedding,
            normalize_embeddings=True
        )
        
        similarities = np.dot(embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print(f"\n   Top 3 Results:")
        for rank, idx in enumerate(top_indices, 1):
            chunk = chunks[idx]
            score = similarities[idx]
            
            q_preview = chunk['arabic_question'][:100].replace('\n', ' ')
            a_preview = chunk['arabic_answer'][:150].replace('\n', ' ')
            
            print(f"\n   {rank}. Score: {score:.4f}")
            print(f"      Fatawa ID: {chunk['fatawa_id']}")
            print(f"      Question: {q_preview}...")
            print(f"      Answer: {a_preview}...")
    
    print(f"\n{'─'*60}")
    print("\n✓ Embedding test completed")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "="*70)
    print("FATAWA EMBEDDING PIPELINE")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract chunks
    chunks = extract_fatawa_chunks(INPUT_JSON)
    
    # Step 2: Preprocess texts
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING ARABIC TEXT")
    print("="*70)
    
    preprocessor = ArabicPreprocessor()
    
    print("\n🔄 Preprocessing fatawa texts...")
    
    for chunk in chunks:
        chunk['question_preprocessed'] = preprocessor.preprocess(chunk['arabic_question'])
        chunk['answer_preprocessed'] = preprocessor.preprocess(chunk['arabic_answer'])
        chunk['combined_preprocessed'] = preprocessor.preprocess(chunk['combined_text'])
    
    # Show preprocessing example
    print("\n📋 Preprocessing Example:")
    print(f"   Original Question (first 100 chars):")
    print(f"   {chunks[0]['arabic_question'][:100]}...")
    print(f"\n   Preprocessed Question (first 100 chars):")
    print(f"   {chunks[0]['question_preprocessed'][:100]}...")
    
    print(f"\n✓ Preprocessed {len(chunks)} chunks")
    
    # Save fatawa_processed.json
    print(f"\n💾 Saving chunks to: {FATAWA_JSON}")
    with open(FATAWA_JSON, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(chunks)} chunks to fatawa_processed.json")
    
    # Step 3: Load model
    model = load_embedding_model(EMBEDDING_MODEL)
    
    # Step 4: Create embeddings (using combined question+answer)
    texts_to_embed = [c['combined_preprocessed'] for c in chunks]
    embeddings = create_embeddings(model, texts_to_embed, BATCH_SIZE)
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()
    
    # Save complete data
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    print(f"\n💾 Saving complete data to: {CHUNKS_PKL}")
    
    save_data = {
        'chunks': chunks,
        'embeddings': embeddings,
        'model_name': EMBEDDING_MODEL,
        'embedding_dim': embeddings.shape[1],
        'num_chunks': len(chunks),
        'created_at': datetime.now().isoformat(),
        'preprocessing': {
            'removed_tashkeel': True,
            'removed_tatweel': True,
            'normalized_arabic': True,
        }
    }
    
    with open(CHUNKS_PKL, 'wb') as f:
        pickle.dump(save_data, f)
    
    file_size = os.path.getsize(CHUNKS_PKL) / (1024 * 1024)
    print(f"✓ Saved to {CHUNKS_PKL} ({file_size:.2f} MB)")
    
    # Step 5: Test embeddings
    test_embeddings(model, chunks, embeddings)
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Total fatawa: {len(chunks)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"\n📁 Output files:")
    print(f"   - {FATAWA_JSON}")
    print(f"   - {CHUNKS_PKL}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
