import os
import json
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Tuple
import faiss
from pathlib import Path
from dotenv import load_dotenv
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAviationRAG:
    def __init__(self):
        """Initialize the simplified Aviation RAG system."""
        load_dotenv()
        
        # Configure Gemini API
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Initialize models
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.embedding_model = 'models/embedding-001'
        
        # Initialize storage
        self.documents = []
        self.embeddings = []
        self.index = None
        
        # Define file paths for persistence
        self.vector_db_path = "aviation_vector_db.faiss"
        self.documents_path = "aviation_documents.json"
        
        # Load or create vector database
        if self.load_existing_database() and self.check_data_freshness():
            logger.info("Loaded existing vector database (data is fresh)")
        else:
            logger.info("Creating new vector database...")
            self.load_aviation_data()
            self.create_embeddings()
            self.build_index()
            self.save_database()
        
        logger.info("RAG system initialized successfully")
    
    def load_existing_database(self) -> bool:
        """Load existing vector database and documents if they exist."""
        try:
            # Check if both files exist
            if not (Path(self.vector_db_path).exists() and Path(self.documents_path).exists()):
                return False
            
            # Load documents
            with open(self.documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            # Load FAISS index
            self.index = faiss.read_index(self.vector_db_path)
            
            # Verify the index matches the documents
            if self.index.ntotal != len(self.documents):
                logger.warning(f"Index size ({self.index.ntotal}) doesn't match documents ({len(self.documents)})")
                return False
            
            # Recreate embeddings array for consistency (small overhead but ensures compatibility)
            self.embeddings = np.zeros((len(self.documents), 768), dtype=np.float32)
            for i in range(min(len(self.documents), self.index.ntotal)):
                vector = self.index.reconstruct(i)
                self.embeddings[i] = vector
            
            logger.info(f"Loaded existing database: {len(self.documents)} documents, {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load existing database: {str(e)}")
            # Clean up corrupted files
            try:
                if Path(self.vector_db_path).exists():
                    Path(self.vector_db_path).unlink()
                if Path(self.documents_path).exists():
                    Path(self.documents_path).unlink()
                logger.info("Cleaned up corrupted database files")
            except:
                pass
            return False
    
    def save_database(self):
        """Save the vector database and documents to disk."""
        try:
            # Save documents
            with open(self.documents_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2)
            
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, self.vector_db_path)
            
            logger.info(f"Saved vector database: {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to save database: {str(e)}")
    
    def check_data_freshness(self) -> bool:
        """Check if the source data has been updated since last indexing."""
        try:
            if not Path(self.documents_path).exists():
                return False
            
            # Get timestamp of saved database
            db_timestamp = Path(self.documents_path).stat().st_mtime
            
            # Check if any source files are newer
            data_dir = Path("aviation_data")
            for file_path in data_dir.rglob("*.txt"):
                if file_path.stat().st_mtime > db_timestamp:
                    logger.info(f"Source file {file_path.name} has been updated")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking data freshness: {str(e)}")
            return False
    
    def rebuild_database(self):
        """Force rebuild the database from source files."""
        logger.info("Rebuilding vector database from source files...")
        
        # Clear existing data
        self.documents = []
        self.embeddings = []
        self.index = None
        
        # Rebuild
        self.load_aviation_data()
        self.create_embeddings()
        self.build_index()
        self.save_database()
        
        logger.info("Database rebuilt successfully")
    
    def load_aviation_data(self):
        """Load aviation text files."""
        logger.info("Loading aviation data...")
        
        data_dir = Path("aviation_data")
        txt_files = list(data_dir.rglob("*.txt"))
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple content extraction - take everything after the header
                lines = content.split('\n')
                content_start = 0
                
                for i, line in enumerate(lines):
                    if line.startswith("="):
                        content_start = i + 1
                        break
                
                main_content = '\n'.join(lines[content_start:]).strip()
                
                if main_content:
                    # Split into smaller chunks
                    chunks = self.simple_chunk_text(main_content, 800)
                    
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 100:  # Only keep substantial chunks
                            doc_info = {
                                'content': chunk.strip(),
                                'source': file_path.name,
                                'chunk_id': i,
                                'file_path': str(file_path)
                            }
                            self.documents.append(doc_info)
                
                logger.info(f"Loaded content from {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        logger.info(f"Total documents loaded: {len(self.documents)}")
    
    def simple_chunk_text(self, text: str, chunk_size: int = 800) -> List[str]:
        """Simple text chunking by character count."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                return np.array(result['embedding'], dtype=np.float32)
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    logger.error(f"Failed to get embedding after {max_retries} attempts")
                    # Return zero vector as fallback
                    return np.zeros(768, dtype=np.float32)
    
    def create_embeddings(self):
        """Create embeddings for all documents."""
        logger.info("Creating embeddings...")
        
        self.embeddings = []
        for i, doc in enumerate(self.documents):
            if i % 5 == 0:
                logger.info(f"Processing embedding {i+1}/{len(self.documents)}")
            
            embedding = self.get_embedding(doc['content'])
            self.embeddings.append(embedding)
            
            # Small delay to be respectful to API
            time.sleep(0.1)
        
        self.embeddings = np.array(self.embeddings)
        logger.info(f"Created {len(self.embeddings)} embeddings")
    
    def build_index(self):
        """Build FAISS index."""
        logger.info("Building FAISS index...")
        
        if len(self.embeddings) == 0:
            logger.warning("No embeddings to index")
            return
        
        # Create FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        
        # Normalize embeddings
        faiss.normalize_L2(self.embeddings)
        
        # Add to index
        self.index.add(self.embeddings)
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents."""
        if self.index is None:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Normalize
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(score)
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer using context."""
        try:
            # Prepare context
            context_text = ""
            for i, doc in enumerate(context_docs):
                context_text += f"\n[Source {i+1}: {doc['source']}]\n"
                context_text += doc['content'][:500] + "...\n"  # Limit context length
                context_text += "-" * 40 + "\n"
            
            # Create prompt
            prompt = f"""You are an aviation expert. Answer the question using the provided context.

CONTEXT:
{context_text}

QUESTION: {query}

Provide a clear, accurate answer based on the context. If the context doesn't contain enough information, say so.

ANSWER:"""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return f"I apologize, but I encountered an error while generating the response: {str(e)}"
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """Main query method."""
        logger.info(f"Processing query: {question}")
        
        try:
            # Search for relevant documents
            relevant_docs = self.search_documents(question, top_k=top_k)
            
            # Generate response
            answer = self.generate_answer(question, relevant_docs)
            
            return {
                'question': question,
                'answer': answer,
                'sources': relevant_docs,
                'num_sources': len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {
                'question': question,
                'answer': f"I apologize, but I encountered an error: {str(e)}",
                'sources': [],
                'num_sources': 0
            }

# Test the system
if __name__ == "__main__":
    try:
        print("🚀 Testing Simple Aviation RAG...")
        rag = SimpleAviationRAG()
        
        # Test query
        result = rag.query("What is ETOPS?")
        
        print(f"\nQuestion: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources used: {result['num_sources']}")
        
        for i, source in enumerate(result['sources']):
            print(f"  {i+1}. {source['source']} (score: {source['similarity_score']:.3f})")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
