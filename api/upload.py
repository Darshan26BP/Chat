from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv
import uuid
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["*"])

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'mite-website')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')  # Change this!

# Initialize OpenAI
openai.api_key = 'OPENAI_API_KEY'

# Initialize Qdrant
try:
    if QDRANT_API_KEY:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
    else:
        client = QdrantClient(url=QDRANT_URL)
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    client = None

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
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=Distance.COSINE
                )
            )
            return True, f"Collection {COLLECTION_NAME} created successfully!"
        else:
            return True, f"Collection {COLLECTION_NAME} already exists."
    except Exception as e:
        return False, f"Error creating collection: {e}"

@app.route('/api/upload/status', methods=['GET'])
def upload_status():
    """Check upload system status"""
    if not client:
        return jsonify({'error': 'Qdrant not connected'}), 500
    
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        collection_exists = COLLECTION_NAME in collection_names
        
        vector_count = 0
        if collection_exists:
            try:
                collection_info = client.get_collection(COLLECTION_NAME)
                vector_count = collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 0
            except:
                vector_count = 0
        
        return jsonify({
            'status': 'ready',
            'collection_exists': collection_exists,
            'collection_name': COLLECTION_NAME,
            'vector_count': vector_count,
            'openai_connected': bool(OPENAI_API_KEY),
            'qdrant_connected': bool(client)
        })
    except Exception as e:
        return jsonify({'error': f'Status check failed: {e}'}), 500

@app.route('/api/upload/data', methods=['POST'])
def upload_data():
    """Upload text data to Qdrant"""
    try:
        data = request.get_json()
        
        # Simple authentication
        if not data or data.get('password') != ADMIN_PASSWORD:
            return jsonify({'error': 'Invalid password'}), 401
        
        text_data = data.get('text_data', '')
        if not text_data:
            return jsonify({'error': 'No text data provided'}), 400
        
        # Create collection if needed
        success, message = create_collection()
        if not success:
            return jsonify({'error': message}), 500
        
        # Process text into chunks
        chunks = []
        if '\n\n---\n\n' in text_data:
            chunks = text_data.split('\n\n---\n\n')
        elif '\n\n' in text_data:
            chunks = text_data.split('\n\n')
        else:
            # Split into smaller chunks
            chunk_size = 1000
            chunks = [text_data[i:i+chunk_size] for i in range(0, len(text_data), chunk_size)]
        
        # Filter chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) > 50]
        
        if not chunks:
            return jsonify({'error': 'No valid chunks found'}), 400
        
        # Upload chunks
        points = []
        successful_uploads = 0
        
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if embedding is None:
                continue
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "chunk_id": i,
                    "source": "web-upload"
                }
            )
            points.append(point)
            
            # Upload in batches of 50
            if len(points) >= 50:
                try:
                    client.upsert(collection_name=COLLECTION_NAME, points=points)
                    successful_uploads += len(points)
                    points = []
                except Exception as e:
                    print(f"Batch upload error: {e}")
        
        # Upload remaining points
        if points:
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                successful_uploads += len(points)
            except Exception as e:
                print(f"Final batch upload error: {e}")
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {successful_uploads} chunks',
            'total_chunks': len(chunks),
            'uploaded_chunks': successful_uploads
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {e}'}), 500

@app.route('/api/upload/clear', methods=['POST'])
def clear_collection():
    """Clear all data from collection"""
    try:
        data = request.get_json()
        
        # Simple authentication
        if not data or data.get('password') != ADMIN_PASSWORD:
            return jsonify({'error': 'Invalid password'}), 401
        
        if not client:
            return jsonify({'error': 'Qdrant not connected'}), 500
        
        try:
            # Delete and recreate collection
            client.delete_collection(COLLECTION_NAME)
            success, message = create_collection()
            
            if success:
                return jsonify({'success': True, 'message': 'Collection cleared successfully'})
            else:
                return jsonify({'error': message}), 500
        except Exception as e:
            return jsonify({'error': f'Clear failed: {e}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Request failed: {e}'}), 500

@app.route('/api/upload/test', methods=['GET'])
def test_search():
    """Test search functionality"""
    try:
        if not client:
            return jsonify({'error': 'Qdrant not connected'}), 500
        
        # Test with a simple query
        test_query = "What is MITE?"
        embedding = get_embedding(test_query)
        
        if not embedding:
            return jsonify({'error': 'Failed to generate test embedding'}), 500
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=3
        )
        
        return jsonify({
            'success': True,
            'query': test_query,
            'results_count': len(results),
            'top_score': results[0].score if results else 0,
            'sample_result': results[0].payload.get('text', '')[:200] + '...' if results else 'No results'
        })
        
    except Exception as e:
        return jsonify({'error': f'Test failed: {e}'}), 500

# For Vercel deployment
if __name__ != '__main__':
    pass
