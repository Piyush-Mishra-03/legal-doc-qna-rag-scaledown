# main.py - Complete FastAPI Backend for Legal Document QnA
# FINAL VERSION - Using direct model loading (works with any transformers version)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any
from pathlib import Path
import aiofiles
import uuid
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Your custom imports
from ingest_pipeline import ingest_pdf_for_chunks
from vector_store import get_all_chunks, COLLECTION_NAME

# Alternative: Use AutoModel directly instead of pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ============================================
# CONFIGURATION
# ============================================

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize embedding model (load once at startup)
print("🔄 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("🔄 Loading FLAN-T5 model and tokenizer...")
# Direct model loading - more reliable than pipeline
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Set device
device = torch.device("cpu")  # Use "cuda" if you have GPU
qa_model = qa_model.to(device)
qa_model.eval()  # Set to evaluation mode

print("✅ LLM loaded!")
print("✅ Embedding model loaded!")

# ============================================
# PYDANTIC MODELS
# ============================================

class ChatRequest(BaseModel):
    question: str

class UploadResponse(BaseModel):
    uploaded: int
    failed: int
    files: List[dict]
    errors: List[dict]

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[dict]
    precedents: List[dict]

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================

app = FastAPI(
    title="Legal Document QnA Backend",
    description="AI-powered Q&A system for legal documents",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_embedding(text: str) -> np.ndarray:
    """
    Generate embedding vector for given text
    
    Args:
        text: Input text string
    
    Returns:
        numpy array of embedding vector
    """
    if not text or text.strip() == "":
        return np.zeros(384)  # Return zero vector for empty text
    
    try:
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(384)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
    
    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Reshape to 2D arrays for sklearn
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        
        similarity = sklearn_cosine_similarity(vec1, vec2)[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def generate_answer(prompt: str, max_length: int = 512) -> str:
    """
    Generate answer using FLAN-T5 model
    
    Args:
        prompt: Input prompt
        max_length: Maximum length of generated answer
    
    Returns:
        Generated answer string
    """
    try:
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        # Generate answer
        with torch.no_grad():
                   outputs = qa_model.generate(
                      **inputs,
        max_length=512,          # Keep this
        min_length=50,           # Change from 20 to 50
        num_beams=4,            # Keep this
        early_stopping=True,     # Keep this
        no_repeat_ngram_size=3,  # Keep this
        temperature=0.7,         # ADD THIS - makes output more creative
        do_sample=True           # ADD THIS - enables sampling
    )
        
        # Decode output
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Legal Document QnA API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": embedding_model is not None,
        "upload_dir": str(UPLOAD_DIR),
        "collection": COLLECTION_NAME
    }

@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process PDF documents
    
    Args:
        files: List of PDF files to upload
    
    Returns:
        Upload summary with success and error details
    """
    saved_files = []
    errors = []

    for file in files:
        # Validate file has a filename
        if not file.filename:
            errors.append({
                "fileName": "Unknown",
                "error": "File has no filename",
                "status": "failed"
            })
            continue
        
        # Validate file type - safely check extension
        if not file.filename.lower().endswith('.pdf'):
            errors.append({
                "fileName": file.filename,
                "error": "Only PDF files are allowed",
                "status": "failed"
            })
            continue

        file_path = None  # Initialize to avoid reference errors
        
        try:
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

            # Save file to disk
            async with aiofiles.open(file_path, "wb") as out:
                content = await file.read()
                await out.write(content)

            print(f"📄 Saved file: {file.filename} ({file_path.stat().st_size} bytes)")

            # Process PDF and ingest chunks into vector store
            print(f"🔄 Processing PDF: {file.filename}")
            chunks = ingest_pdf_for_chunks(file_path)
            
            print(f"✅ Processed {len(chunks)} chunks from {file.filename}")
            
            saved_files.append({
                "fileName": file.filename,
                "storedAs": file_path.name,
                "fileId": file_id,
                "chunks": len(chunks),
                "status": "success",
                "size": file_path.stat().st_size
            })
            
        except Exception as e:
            print(f"❌ Error processing {file.filename}: {e}")
            import traceback
            traceback.print_exc()  # Print full error trace for debugging
            
            errors.append({
                "fileName": file.filename,
                "error": str(e),
                "status": "failed"
            })
            
            # Clean up failed file
            try:
                if file_path and file_path.exists():
                    file_path.unlink()
                    print(f"🗑️ Cleaned up failed file: {file.filename}")
            except Exception as cleanup_error:
                print(f"⚠️ Could not clean up file {file.filename}: {cleanup_error}")

    return UploadResponse(
        uploaded=len(saved_files),
        failed=len(errors),
        files=saved_files,
        errors=errors
    )
@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Answer questions based on uploaded documents using RAG (Retrieval Augmented Generation)
    
    Args:
        req: ChatRequest containing the user's question
    
    Returns:
        ChatResponse with answer, confidence, citations, and precedents
    """
    question = req.question
    
    # Validate question
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    print(f"💬 Question: {question}")

    try:
        # Retrieve all chunks from vector store
        chunks = get_all_chunks(limit=1000)
        
        if not chunks:
            return ChatResponse(
                answer="No documents available. Please upload PDF documents first.",
                confidence=0.0,
                citations=[],
                precedents=[]
            )

        print(f"📚 Retrieved {len(chunks)} chunks from vector store")

        # Generate embedding for the question
        q_emb = get_embedding(question)

        # Calculate similarities with all chunks
        similarities = []
        for chunk in chunks:
            chunk_text = chunk.get("compressed_text") or chunk.get("text") or ""
            if not chunk_text or len(chunk_text.strip()) == 0:
                continue
                
            chunk_emb = get_embedding(chunk_text)
            sim = cosine_similarity(q_emb, chunk_emb)
            
            similarities.append({
                "similarity": sim,
                "chunk": chunk
            })

        if not similarities:
            return ChatResponse(
                answer="No valid content found in documents.",
                confidence=0.0,
                citations=[],
                precedents=[]
            )

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        print(f"🔍 Top similarity score: {similarities[0]['similarity']:.4f}")

        # Get top K most relevant chunks
        top_k = 5
        top_chunks = similarities[:top_k]

        # Filter chunks with similarity above threshold
        threshold = 0.2
        relevant_chunks = [c for c in top_chunks if c["similarity"] >= threshold]
        
        if not relevant_chunks:
            return ChatResponse(
                answer="I couldn't find relevant information to answer your question.",
                confidence=0.0,
                citations=[],
                precedents=[]
            )

        # Build context from top chunks
        context_parts = []
        for idx, item in enumerate(relevant_chunks[:3], 1):
            chunk = item["chunk"]
            text = chunk.get("compressed_text") or chunk.get("text")
            
            if text is None:
                continue
            
            text = text.strip()
            if not text:
                continue
            
            # Truncate long text
            truncated = text[:600] + "..." if len(text) > 600 else text
            context_parts.append(truncated)

        if not context_parts:
            context_text = "Relevant information was found, but could not be formatted."
        else:
            context_text = "\n\n".join(context_parts)

        # Build prompt for LLM
        prompt = f"""You are a legal expert assistant. Based on the legal documents below, provide a clear and detailed answer.

Legal Context:
{context_text}

Legal Question: {question}

Provide a comprehensive answer explaining the legal principles, consequences, and relevant provisions. Use 3-4 complete sentences:"""

        # Generate answer with robust error handling
        try:
            print("🤖 Generating answer with FLAN-T5...")
            
            llm_result = generate_answer(prompt, max_length=512)
            
            print(f"📝 LLM raw output: {llm_result}")
            
            # Check if LLM gave a poor answer
            poor_answer_indicators = [
                len(llm_result) < 30,
                'Dr.' in llm_result and 'Sultan' in llm_result,
                llm_result.count('.') < 2,
                'Specific Relief Act' in llm_result and len(llm_result) < 100
            ]
            
            if any(poor_answer_indicators):
                print("⚠️ LLM gave poor answer, using direct context extraction")
                
                answer_parts = []
                for idx, item in enumerate(relevant_chunks[:2], 1):
                    chunk = item["chunk"]
                    text = chunk.get("compressed_text") or chunk.get("text", "")
                    
                    if text:
                        excerpt = text[:300].strip()
                        last_period = excerpt.rfind('.')
                        if last_period > 150:
                            excerpt = excerpt[:last_period + 1]
                        else:
                            excerpt += "..."
                        answer_parts.append(excerpt)
                
                if answer_parts:
                    llm_result = "Based on the legal documents:\n\n" + "\n\n".join(answer_parts)
                else:
                    llm_result = "I found relevant information but couldn't extract a clear answer. Please refer to the citations below."
            
            print(f"✅ Final answer length: {len(llm_result)}")
            
        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}")
            import traceback
            traceback.print_exc()
            llm_result = f"Based on the uploaded legal documents:\n\n{context_text[:500]}..."

        # Calculate average confidence
        confidence = float(np.mean([c["similarity"] for c in relevant_chunks]))

        # Build citations
        citations = []
        for item in relevant_chunks:
            chunk = item["chunk"]
            citations.append({
                "fileName": chunk.get("file_name", "Unknown"),
                "page": chunk.get("page", 0),
                "similarity": round(float(item["similarity"]), 4)
            })

        # Build precedents
        precedents = []
        for item in relevant_chunks:
            chunk = item["chunk"]
            sim_score = item["similarity"]
            
            if sim_score > 0.7:
                relevance = "High"
            elif sim_score > 0.4:
                relevance = "Medium"
            else:
                relevance = "Low"
            
            preview_text = chunk.get("compressed_text") or chunk.get("text") or ""
            precedents.append({
                "title": f"{chunk.get('file_name', 'Unknown')} (Page {chunk.get('page', 0)})",
                "similarity": round(float(sim_score), 4),
                "relevance": relevance,
                "preview": preview_text[:150] + ("..." if len(preview_text) > 150 else "")
            })

        print(f"✅ Generated answer with confidence: {confidence:.4f}")

        return ChatResponse(
            answer=llm_result,
            confidence=round(confidence, 4),
            citations=citations,
            precedents=precedents
        )

    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/api/documents/list")
async def list_documents():
    """
    List all uploaded documents with statistics
    
    Returns:
        List of documents with chunk and page counts
    """
    try:
        chunks = get_all_chunks(limit=10000)
        
        # Aggregate by file name
        files = {}
        for chunk in chunks:
            file_name = chunk.get("file_name")
            if file_name:
                if file_name not in files:
                    files[file_name] = {
                        "fileName": file_name,
                        "chunks": 0,
                        "pages": set()
                    }
                files[file_name]["chunks"] += 1
                page = chunk.get("page")
                if page:
                    files[file_name]["pages"].add(page)
        
        # Convert to list and format
        file_list = []
        for file_name, info in files.items():
            file_list.append({
                "fileName": file_name,
                "chunks": info["chunks"],
                "pages": len(info["pages"])
            })
        
        # Sort by file name
        file_list.sort(key=lambda x: x["fileName"])
        
        return {
            "documents": file_list,
            "total": len(file_list),
            "totalChunks": sum(f["chunks"] for f in file_list)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/api/documents/{file_name}")
async def delete_document(file_name: str):
    """
    Delete a specific document (placeholder - implement based on your needs)
    
    Args:
        file_name: Name of the file to delete
    """
    # TODO: Implement document deletion from vector store
    return {
        "message": f"Delete endpoint for {file_name}",
        "note": "Implementation depends on your vector store setup"
    }

@app.get("/api/stats")
async def get_stats():
    """
    Get system statistics
    """
    try:
        chunks = get_all_chunks(limit=10000)
        
        files = set()
        total_chunks = len(chunks)
        
        for chunk in chunks:
            file_name = chunk.get("file_name")
            if file_name:
                files.add(file_name)
        
        return {
            "totalDocuments": len(files),
            "totalChunks": total_chunks,
            "modelName": "all-MiniLM-L6-v2",
            "embeddingDimension": 384,
            "uploadDirectory": str(UPLOAD_DIR)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# ============================================
# STARTUP & SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("🚀 Starting Legal Document QnA Backend...")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"🤖 Embedding model: all-MiniLM-L6-v2")
    print(f"🗄️ Vector store collection: {COLLECTION_NAME}")
    print("✅ Application ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("👋 Shutting down Legal Document QnA Backend...")

@app.get("/api/debug/vector-store")
async def debug_vector_store():
    """Debug vector store connection and data"""
    try:
        from vector_store import client, COLLECTION_NAME
        
        # Check if Weaviate is ready
        is_ready = client.is_ready()
        
        # Check if collection exists
        collection_exists = client.collections.exists(COLLECTION_NAME)
        
        # Try to get chunks
        chunks = get_all_chunks(limit=10)
        
        return {
            "weaviate_ready": is_ready,
            "collection_name": COLLECTION_NAME,
            "collection_exists": collection_exists,
            "chunks_retrieved": len(chunks),
            "sample_chunks": chunks[:2] if chunks else []
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*50)
    print("🚀 Starting FastAPI server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("="*50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )