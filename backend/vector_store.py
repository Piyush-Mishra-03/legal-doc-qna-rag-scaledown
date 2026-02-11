import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType

# 1️⃣ Connect correctly (v4 syntax)
try:
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051  # Default gRPC port
    )
    print("✅ Connected to Weaviate!")
except Exception as e:
    raise RuntimeError(f"Failed to connect to Weaviate: {e}")

# Check if client is ready
if not client.is_ready():
    raise RuntimeError("Weaviate is not ready")

COLLECTION_NAME = "LegalChunks"

# 2️⃣ Create collection (v4 syntax)
try:
    # Check if collection exists
    if not client.collections.exists(COLLECTION_NAME):
        client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # No vectorization
            properties=[
                Property(name="file_name", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="compressed_text", data_type=DataType.TEXT)
            ]
        )
        print(f"✅ Created collection '{COLLECTION_NAME}'")
    else:
        print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists")
except Exception as e:
    raise RuntimeError(f"Collection creation failed: {e}")

# 3️⃣ Get collection reference
collection = client.collections.get(COLLECTION_NAME)

# 4️⃣ Store chunks (v4 batch syntax)
def store_chunks(chunks, file_name):
    """
    Store document chunks in Weaviate

    chunks come from ingest_pipeline and contain:
    - page
    - compressed_text
    - original_text (fallback)
    """

    if not chunks:
        print("⚠️ No chunks to store")
        return

    stored = 0

    try:
        with collection.batch.dynamic() as batch:
            for idx, c in enumerate(chunks):

                # pick correct text coming from ingest pipeline
                text = c.get("compressed_text") or c.get("original_text")

                if not isinstance(c, dict) or "page" not in c or not text or not text.strip():
                    print(f"⚠️ Skipping invalid chunk at index {idx}")
                    continue

                batch.add_object(
                    properties={
                        "file_name": file_name,
                        "page": int(c["page"]),
                        "compressed_text": str(c["compressed_text"])
                    }
                )

                stored += 1

        print(f"✅ Successfully stored {stored} chunks from '{file_name}'")

    except Exception as e:
        print(f"❌ Error storing chunks: {e}")
        raise


# 5️⃣ Fetch all chunks (v4 query syntax)
def get_all_chunks(limit: int = 1000):
    collection = client.collections.get(COLLECTION_NAME)

    result = collection.query.fetch_objects(
        limit=limit
    )

    chunks = []

    for obj in result.objects:
        props = obj.properties

        chunks.append({
            "compressed_text": props.get("compressed_text"),
            "text": props.get("text"),
            "file_name": props.get("file_name"),
            "page": props.get("page"),
            "chunk_id": props.get("chunk_id")
        })

    return chunks


# 6️⃣ Get chunks by file
def get_chunks_by_file(file_name):
    """Get all chunks from a specific file"""
    try:
        response = collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("file_name").equal(file_name),
            limit=1000
        )
        
        chunks = []
        for obj in response.objects:
            chunks.append({
                "uuid": str(obj.uuid),
                "file_name": obj.properties.get("file_name"),
                "page": obj.properties.get("page"),
                "compressed_text": obj.properties.get("compressed_text")
            })
        
        print(f"✅ Retrieved {len(chunks)} chunks for '{file_name}'")
        return chunks
        
    except Exception as e:
        print(f"❌ Error retrieving chunks: {e}")
        return []

# 7️⃣ Count chunks
def count_chunks():
    """Count total number of chunks"""
    try:
        response = collection.aggregate.over_all(total_count=True)
        count = response.total_count
        print(f"📊 Total chunks: {count}")
        return count
    except Exception as e:
        print(f"❌ Error counting: {e}")
        return 0

# 8️⃣ Delete all chunks
def delete_all_chunks():
    """Delete all objects from collection"""
    try:
        collection.data.delete_many(
            where=wvc.query.Filter.by_property("file_name").like("*")
        )
        print(f"🗑️ Deleted all chunks")
    except Exception as e:
        print(f"❌ Error deleting: {e}")

# 9️⃣ Close connection when done
def close_connection():
    client.close()
    print("🔌 Connection closed")

# 🔟 Example usage
if __name__ == "__main__":
    try:
        # Test data
        test_chunks = [
            {"page": 1, "text": "First page content"},
            {"page": 2, "text": "Second page content"},
            {"page": 3, "text": "Third page content"}
        ]
        
        # Store chunks
        store_chunks(test_chunks, "test_doc.pdf")
        
        # Count
        count_chunks()
        
        # Retrieve all
        all_chunks = get_all_chunks()
        print(f"\n📄 All chunks: {all_chunks[:2]}")  # Print first 2
        
        # Get by file
        file_chunks = get_chunks_by_file("test_doc.pdf")
        print(f"\n📄 File chunks: {file_chunks}")
        
    finally:
        # Always close connection
        close_connection()