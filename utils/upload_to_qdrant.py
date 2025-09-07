import os
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import uuid
import json

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'mite-website')

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Qdrant
if QDRANT_API_KEY:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
else:
    client = QdrantClient(url=QDRANT_URL)

def get_embedding(text):
    """Generate embedding using OpenAI's embedding model"""
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def create_collection():
    """Create Qdrant collection if it doesn't exist"""
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            print(f"Creating collection: {COLLECTION_NAME}")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=Distance.COSINE
                )
            )
            print(f"Collection {COLLECTION_NAME} created successfully!")
        else:
            print(f"Collection {COLLECTION_NAME} already exists.")
            
    except Exception as e:
        print(f"Error creating collection: {e}")
        return False
    
    return True

def upload_chunks_to_qdrant(chunks, batch_size=50):
    """Upload text chunks to Qdrant with embeddings"""
    print(f"Starting upload of {len(chunks)} chunks to Qdrant...")
    
    points = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Generate embedding
        embedding = get_embedding(chunk)
        if embedding is None:
            print(f"Skipping chunk {i+1} due to embedding error")
            continue
        
        # Create point
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk,
                "chunk_id": i,
                "source": "mite-website"
            }
        )
        points.append(point)
        
        # Upload in batches
        if len(points) >= batch_size:
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                print(f"Uploaded batch of {len(points)} points")
                points = []
            except Exception as e:
                print(f"Error uploading batch: {e}")
                points = []
    
    # Upload remaining points
    if points:
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            print(f"Uploaded final batch of {len(points)} points")
        except Exception as e:
            print(f"Error uploading final batch: {e}")

def load_text_chunks(file_path):
    """Load text chunks from file"""
    chunks = []
    
    if file_path.endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by double newlines or custom delimiter
                if '\n\n---\n\n' in content:
                    chunks = content.split('\n\n---\n\n')
                elif '\n\n' in content:
                    chunks = content.split('\n\n')
                else:
                    # Split into smaller chunks if no delimiters
                    chunk_size = 1000
                    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        except Exception as e:
            print(f"Error reading text file: {e}")
            
    elif file_path.endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    chunks = [item if isinstance(item, str) else str(item) for item in data]
                elif isinstance(data, dict):
                    chunks = [str(value) for value in data.values() if isinstance(value, str)]
        except Exception as e:
            print(f"Error reading JSON file: {e}")
    
    # Filter out empty chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) > 50]
    
    return chunks

def main():
    print("MITE Website Content Upload to Qdrant")
    print("=" * 40)
    
    # Create collection
    if not create_collection():
        print("Failed to create collection. Exiting.")
        return
    
    # Load chunks from your existing files
    chunk_files = [
        '../Client/mite_website_chunks.txt',
        '../Client/mite_website_full_content.txt'
    ]
    
    all_chunks = []
    
    for file_path in chunk_files:
        if os.path.exists(file_path):
            print(f"Loading chunks from: {file_path}")
            chunks = load_text_chunks(file_path)
            all_chunks.extend(chunks)
            print(f"Loaded {len(chunks)} chunks from {file_path}")
        else:
            print(f"File not found: {file_path}")
    
    if not all_chunks:
        print("No chunks to upload. Please check your data files.")
        return
    
    print(f"\nTotal chunks to upload: {len(all_chunks)}")
    
    # Remove duplicates
    all_chunks = list(set(all_chunks))
    print(f"Unique chunks after deduplication: {len(all_chunks)}")
    
    # Upload to Qdrant
    upload_chunks_to_qdrant(all_chunks)
    
    # Verify upload
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"\nUpload completed!")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Total vectors: {collection_info.vectors_count}")
        
        # Test search
        print("\nTesting search functionality...")
        test_query = "What is MITE?"
        test_embedding = get_embedding(test_query)
        
        if test_embedding:
            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=test_embedding,
                limit=3
            )
            print(f"Test search returned {len(results)} results")
            if results:
                print(f"Top result score: {results[0].score}")
        
    except Exception as e:
        print(f"Error verifying upload: {e}")

if __name__ == "__main__":
    main()
