from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import pinecone
import os
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp-free')
)

# Connect to your Pinecone index
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'mite-website-index')
try:
    index = pinecone.Index(INDEX_NAME)
except:
    index = None

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

def search_similar_chunks(query, top_k=5):
    """Search for similar text chunks in Pinecone"""
    if not index:
        return []
    
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [match['metadata']['text'] for match in results['matches']]
    except Exception as e:
        print(f"Error searching chunks: {e}")
        return []

def generate_response(query, context_chunks):
    """Generate response using OpenAI GPT with context"""
    try:
        context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        
Context:
{context}

Question: {query}

Instructions:
- Answer the question based on the provided context
- If the context doesn't contain relevant information, say so politely
- Be concise but informative
- Maintain a friendly, professional tone

Answer:"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm sorry, I encountered an error while processing your question. Please try again."

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Search for relevant context
        context_chunks = search_similar_chunks(user_message)
        
        # Generate response
        response = generate_response(user_message, context_chunks)
        
        return jsonify({
            'response': response,
            'sources_used': len(context_chunks)
        })
        
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    qdrant_status = "connected" if client else "disconnected"
    
    # Try to get collection info
    collection_info = None
    if client:
        try:
            collection_info = client.get_collection(COLLECTION_NAME)
            vector_count = collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else "unknown"
        except Exception as e:
            vector_count = f"error: {str(e)}"
    else:
        vector_count = "no connection"
    
    return jsonify({
        'status': 'healthy', 
        'service': 'RAG Chatbot API',
        'qdrant_status': qdrant_status,
        'collection': COLLECTION_NAME,
        'vector_count': vector_count
    })

# Vercel serverless function handler
def handler(request):
    return app(request.environ, lambda status, headers: None)
