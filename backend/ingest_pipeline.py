# ingest_pipeline.py - Fixed PDF Processing Pipeline

from pathlib import Path
from typing import List, Dict, Union
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Your custom imports
from scaledown_client import compress_text, estimate_tokens
from vector_store import store_chunks

def extract_pages(pdf_path: Union[Path, str]) -> List[Dict]:
    """
    Extract text from each page of a PDF
    
    Args:
        pdf_path: Path to PDF file (can be Path object or string)
    
    Returns:
        List of dictionaries with page number and text
    """
    # Convert to Path if string
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)
    
    # Validate file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Validate file is PDF
    if pdf_path.suffix.lower() != '.pdf':
        raise ValueError(f"File must be a PDF: {pdf_path}")
    
    try:
        # Open PDF document
        doc = fitz.open(str(pdf_path))  # Convert Path to string for fitz.open()
        pages = []
        
        # Iterate through pages
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            # Extract text using get_text() - returns string in PyMuPDF 1.26.7
            text = page.get_text()
            
            # Safety: ensure text is string and not None
            if text is not None and isinstance(text, str) and text.strip():
                pages.append({
                    "page": page_num + 1,  # 1-indexed page numbers
                    "text": text.strip()
                })
            elif text:  # Fallback for unexpected types
                text_str = str(text).strip()
                if text_str:
                    pages.append({
                        "page": page_num + 1,
                        "text": text_str
                    })
        
        # Close document
        doc.close()
        
        print(f"✅ Extracted {len(pages)} pages from {pdf_path.name}")
        return pages
        
    except Exception as e:
        print(f"❌ Error extracting pages from {pdf_path.name}: {e}")
        raise

def chunk_pages(pages: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Dict]:
    """
    Split pages into smaller chunks with overlap
    
    Args:
        pages: List of page dictionaries with 'page' and 'text' keys
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
    
    Returns:
        List of chunk dictionaries with compression metadata
    """
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = []
    total_original_tokens = 0
    total_compressed_tokens = 0
    
    for p in pages:
        # Split page text into chunks
        split_texts = splitter.split_text(p["text"])
        
        for idx, t in enumerate(split_texts):
            try:
                # Calculate token counts
                original_tokens = estimate_tokens(t)
                
                # Compress text
                compressed_text = compress_text(t)
                compressed_tokens = estimate_tokens(compressed_text)
                
                # Track totals
                total_original_tokens += original_tokens
                total_compressed_tokens += compressed_tokens
                
                # Create chunk object
                chunks.append({
                    "page": p["page"],
                    "chunk_index": idx,
                    "text": t,
                    "compressed_text": compressed_text,
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "compression_ratio": round(compressed_tokens / original_tokens, 2) if original_tokens > 0 else 1.0
                })
                
            except Exception as e:
                print(f"⚠️ Error processing chunk {idx} on page {p['page']}: {e}")
                # Add uncompressed chunk as fallback
                chunks.append({
                    "page": p["page"],
                    "chunk_index": idx,
                    "original_text": t,
                    "compressed_text": t,  # Use original if compression fails
                    "original_tokens": len(t.split()),
                    "compressed_tokens": len(t.split()),
                    "compression_ratio": 1.0
                })
    
    # Print compression statistics
    if total_original_tokens > 0:
        overall_ratio = total_compressed_tokens / total_original_tokens
        print(f"📊 Compression stats:")
        print(f"   Original tokens: {total_original_tokens}")
        print(f"   Compressed tokens: {total_compressed_tokens}")
        print(f"   Compression ratio: {overall_ratio:.2%}")
    
    print(f"✅ Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks

def ingest_pdf_for_chunks(pdf_path: Union[Path, str]) -> List[Dict]:
    """
    Complete pipeline: Extract pages, chunk, compress, and store in vector DB
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        List of processed chunks
    """
    # Convert to Path if string
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)
    
    print(f"🔄 Processing PDF: {pdf_path.name}")
    
    try:
        # Step 1: Extract pages
        pages = extract_pages(pdf_path)
        
        if not pages:
            raise ValueError(f"No text content found in PDF: {pdf_path.name}")
        
        # Step 2: Chunk and compress
        chunks = chunk_pages(pages)
        
        if not chunks:
            raise ValueError(f"No chunks created from PDF: {pdf_path.name}")
        
        # Step 3: Store in vector database
        store_chunks(chunks, pdf_path.name)
        
        print(f"✅ Successfully ingested {pdf_path.name}")
        print(f"   Pages: {len(pages)}")
        print(f"   Chunks: {len(chunks)}")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error ingesting PDF {pdf_path.name}: {e}")
        raise

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_pdf_metadata(pdf_path: Union[Path, str]) -> Dict:
    """
    Get metadata from PDF file
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Dictionary with PDF metadata
    """
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)
    
    try:
        doc = fitz.open(str(pdf_path))
        
        # Get metadata - handle None case
        metadata_dict = doc.metadata if doc.metadata else {}
        
        metadata = {
            "title": metadata_dict.get("title", "Unknown") if metadata_dict else "Unknown",
            "author": metadata_dict.get("author", "Unknown") if metadata_dict else "Unknown",
            "subject": metadata_dict.get("subject", "") if metadata_dict else "",
            "pages": doc.page_count,
            "file_size": pdf_path.stat().st_size,
            "file_name": pdf_path.name
        }
        doc.close()
        return metadata
    except Exception as e:
        print(f"Error getting metadata: {e}")
        return {
            "error": str(e),
            "file_name": pdf_path.name if pdf_path else "Unknown"
        }

def validate_pdf(pdf_path: Union[Path, str]) -> bool:
    """
    Validate if file is a valid PDF
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        True if valid PDF, False otherwise
    """
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"File does not exist: {pdf_path}")
        return False
    
    if pdf_path.suffix.lower() != '.pdf':
        print(f"File is not a PDF: {pdf_path}")
        return False
    
    try:
        doc = fitz.open(pdf_path)
        is_valid = len(doc) > 0
        doc.close()
        return is_valid
    except Exception as e:
        print(f"Invalid PDF file: {e}")
        return False

# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    # Test the pipeline
    test_pdf = Path("test_document.pdf")
    
    if test_pdf.exists():
        print("="*50)
        print("Testing PDF Ingestion Pipeline")
        print("="*50)
        
        # Get metadata
        metadata = get_pdf_metadata(test_pdf)
        print(f"\n📄 Metadata: {metadata}")
        
        # Validate
        is_valid = validate_pdf(test_pdf)
        print(f"✅ Valid PDF: {is_valid}")
        
        # Process
        if is_valid:
            chunks = ingest_pdf_for_chunks(test_pdf)
            print(f"\n✅ Total chunks created: {len(chunks)}")
    else:
        print("⚠️ test_document.pdf not found")