"""
Hadith Shar7 Embedding Pipeline
===============================
This script processes hadith explanations (shar7) for semantic search using embeddings.

Best Embedding Models for Arabic/Islamic Content:
-------------------------------------------------
1. intfloat/multilingual-e5-large (RECOMMENDED)
   - 1024 dimensions, excellent Arabic support
   - Trained on diverse multilingual data
   - Best semantic understanding for religious texts

2. sentence-transformers/paraphrase-multilingual-mpnet-base-v2
   - 768 dimensions, good multilingual support
   - Faster inference, smaller model

3. aubmindlab/bert-base-arabertv2
   - Arabic-specific, trained on Arabic Wikipedia + news
   - Good for classical Arabic but less semantic search optimized

We use multilingual-e5-large for best results with Islamic/Arabic scholarly text.

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
INPUT_JSON = "hadith_shar7_combined.json"  # Combined hadith JSON
OUTPUT_DIR = Path(__file__).parent
SHAR7_JSON = OUTPUT_DIR / "shar7.json"
CHUNKS_PKL = OUTPUT_DIR / "shar7_chunks_complete.pkl"

# Embedding model - Best for Arabic/Islamic content
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
# Alternative: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Batch size for embedding (adjust based on GPU memory)
BATCH_SIZE = 32

# ============================================================================
# ARABIC TEXT PREPROCESSING
# ============================================================================

class ArabicPreprocessor:
    """
    Arabic text preprocessing pipeline optimized for Islamic scholarly texts.
    
    Key considerations:
    - Preserve meaning while normalizing variations
    - Keep Islamic honorifics and phrases
    - Remove diacritics for better matching
    - Handle classical Arabic variations
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
        
        # Common Islamic phrases to preserve (for reference)
        self.islamic_phrases = [
            'صلى الله عليه وسلم',
            'رضي الله عنه',
            'رضي الله عنها',
            'رحمه الله',
            'عز وجل',
            'سبحانه وتعالى',
        ]
    
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
        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_non_arabic_noise(self, text: str) -> str:
        """Remove URLs, emails, and excessive punctuation."""
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove repeated punctuation
        text = re.sub(r'([.!?،؛])\1+', r'\1', text)
        return text
    
    def preprocess(self, text: str, keep_tashkeel: bool = False) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Input Arabic text
            keep_tashkeel: If True, preserve diacritics (useful for Quran)
        
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Step 1: Remove noise
        text = self.remove_non_arabic_noise(text)
        
        # Step 2: Remove tatweel
        text = self.remove_tatweel(text)
        
        # Step 3: Remove tashkeel (optional)
        if not keep_tashkeel:
            text = self.remove_tashkeel(text)
        
        # Step 4: Normalize Arabic characters
        text = self.normalize_arabic_chars(text)
        
        # Step 5: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        return text

# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_shar7_chunks(input_path: str) -> list:
    """
    Extract shar7 (explanations) with metadata from combined JSON.
    
    Each chunk contains:
    - shar7_text: The explanation text
    - hadith_text: Original hadith
    - metadata: rawi, mohdith, book, grade, sharhId
    """
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING SHAR7 CHUNKS")
    print("="*70)
    
    print(f"\n📂 Loading data from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} hadith entries")
    
    chunks = []
    skipped = 0
    
    for item in data:
        shar7_text = item.get('sharh', {}).get('text', '')
        
        # Skip empty shar7
        if not shar7_text or len(shar7_text.strip()) < 10:
            skipped += 1
            continue
        
        chunk = {
            'sharhId': item.get('sharhId'),
            'shar7_text': shar7_text,
            'hadith_text': item.get('hadith', {}).get('text', ''),
            'metadata': {
                'rawi': item.get('hadith', {}).get('rawi', ''),
                'mohdith': item.get('hadith', {}).get('mohdith', ''),
                'book': item.get('hadith', {}).get('book', ''),
                'numberOrPage': item.get('hadith', {}).get('numberOrPage', ''),
                'grade': item.get('hadith', {}).get('grade', ''),
            }
        }
        chunks.append(chunk)
    
    print(f"✓ Extracted {len(chunks)} valid shar7 chunks")
    print(f"⚠ Skipped {skipped} entries (empty or too short)")
    
    # Statistics
    text_lengths = [len(c['shar7_text']) for c in chunks]
    print(f"\n📊 Shar7 Text Statistics:")
    print(f"   Min length: {min(text_lengths)} chars")
    print(f"   Max length: {max(text_lengths)} chars")
    print(f"   Avg length: {sum(text_lengths)//len(text_lengths)} chars")
    
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
    For passages (our shar7 texts), we use "passage: " prefix.
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
    
    # Test queries - Islamic/Hadith related
    test_queries = [
        "ما حكم لبس الحرير للرجال",  # Ruling on silk for men
        "آداب العيد في الإسلام",  # Eid etiquette in Islam
        "الأكل قبل صلاة العيد",  # Eating before Eid prayer
        "معاملة الزوجة بالرفق",  # Treating wife with kindness
        "اللهو المباح في الإسلام",  # Permissible entertainment in Islam
    ]
    
    print(f"\n🔍 Testing with {len(test_queries)} queries...\n")
    
    # Preprocess queries
    preprocessor = ArabicPreprocessor()
    
    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"📝 Query: {query}")
        
        # Preprocess query
        processed_query = preprocessor.preprocess(query)
        
        # For E5 models, add query prefix
        if 'e5' in EMBEDDING_MODEL.lower():
            query_for_embedding = f"query: {processed_query}"
        else:
            query_for_embedding = processed_query
        
        # Get query embedding
        query_embedding = model.encode(
            query_for_embedding,
            normalize_embeddings=True
        )
        
        # Calculate cosine similarities
        similarities = np.dot(embeddings, query_embedding)
        
        # Get top 3 results
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print(f"\n   Top 3 Results:")
        for rank, idx in enumerate(top_indices, 1):
            chunk = chunks[idx]
            score = similarities[idx]
            
            # Truncate shar7 for display
            shar7_preview = chunk['shar7_text'][:200].replace('\n', ' ')
            
            print(f"\n   {rank}. Score: {score:.4f}")
            print(f"      Hadith: {chunk['hadith_text'][:100]}...")
            print(f"      Book: {chunk['metadata']['book']}")
            print(f"      Shar7: {shar7_preview}...")
    
    print(f"\n{'─'*60}")
    print("\n✓ Embedding test completed")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "="*70)
    print("HADITH SHAR7 EMBEDDING PIPELINE")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract chunks
    chunks = extract_shar7_chunks(INPUT_JSON)
    
    # Step 2: Preprocess texts
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING ARABIC TEXT")
    print("="*70)
    
    preprocessor = ArabicPreprocessor()
    
    print("\n🔄 Preprocessing shar7 texts...")
    
    for chunk in chunks:
        chunk['shar7_preprocessed'] = preprocessor.preprocess(chunk['shar7_text'])
        chunk['hadith_preprocessed'] = preprocessor.preprocess(chunk['hadith_text'])
    
    # Show preprocessing example
    print("\n📋 Preprocessing Example:")
    print(f"   Original (first 100 chars):")
    print(f"   {chunks[0]['shar7_text'][:100]}...")
    print(f"\n   Preprocessed (first 100 chars):")
    print(f"   {chunks[0]['shar7_preprocessed'][:100]}...")
    
    print(f"\n✓ Preprocessed {len(chunks)} chunks")
    
    # Save shar7.json
    print(f"\n💾 Saving chunks to: {SHAR7_JSON}")
    with open(SHAR7_JSON, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(chunks)} chunks to shar7.json")
    
    # Step 3: Load model
    model = load_embedding_model(EMBEDDING_MODEL)
    
    # Step 4: Create embeddings
    texts_to_embed = [c['shar7_preprocessed'] for c in chunks]
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
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"\n📁 Output files:")
    print(f"   - {SHAR7_JSON}")
    print(f"   - {CHUNKS_PKL}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
