# Aviation Knowledge Assistant

A sophisticated AI-powered chatbot specialized in aviation knowledge, built with RAG (Retrieval-Augmented Generation) technology.

## Features

- 🛩️ **Aviation Expertise**: Comprehensive knowledge about flight operations, safety procedures, regulations, and technical concepts
- 🔍 **Source Attribution**: Every answer includes relevant source documents with relevance scores
- ⚡ **Real-time Streaming**: Responses stream in real-time for better user experience
- 💾 **Persistent Knowledge Base**: Fast loading with pre-built vector database
- 🎯 **Smart Search**: Semantic search across 123+ aviation document chunks
- 📱 **Clean Interface**: Intuitive Streamlit web interface with collapsible source viewing

## Knowledge Base

The system is trained on authoritative aviation documents including:
- ETOPS (Extended Twin-engine Operational Performance Standards)
- Flight Planning procedures
- MEL (Minimum Equipment List) guidelines
- IATA delay codes
- Airport Collaborative Decision Making (A-CDM)
- Flight dispatcher responsibilities
- Aircraft fuel economy principles
- Mach number effects on flight operations

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**:
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

4. **Run the application**:
   ```bash
   streamlit run streaming_app.py
   ```

5. **Open your browser** to `http://localhost:8502`

## Usage

### Sample Questions
Try asking about:
- "What is ETOPS and why is it critical for twin-engine aircraft operations?"
- "How does flight planning work for commercial aviation?"
- "What are IATA delay codes and how are they used?"
- "What is the role of a flight dispatcher?"
- "What is MEL and when is it used?"

### Interface Features
- **Sample Questions**: Click any preset question in the sidebar for instant answers
- **Source Viewing**: Expand "View Sources" to see where information comes from
- **Settings**: Adjust number of sources and enable/disable streaming
- **Debug Mode**: Enable for detailed processing information

## Technical Architecture

### Core Components
- **RAG System** (`simple_rag.py`): Handles document retrieval and response generation
- **Web Interface** (`streaming_app.py`): Streamlit-based user interface
- **Vector Database**: Pre-built FAISS index for fast semantic search
- **Knowledge Base**: 9 curated aviation documents with proper attribution

### Key Technologies
- **LLM**: Google Gemini 1.5 Flash for response generation
- **Embeddings**: Google's embedding-001 model
- **Vector Search**: FAISS for similarity search
- **Frontend**: Streamlit for web interface
- **Data Processing**: BeautifulSoup for document processing

## File Structure

```
aviation-rag/
├── streaming_app.py          # Main Streamlit application
├── simple_rag.py             # RAG system core
├── manage_db.py              # Database management utilities
├── requirements.txt          # Python dependencies
├── .env                      # API keys (create this)
├── aviation_data/            # Source documents (9 files)
├── aviation_vector_db.faiss  # Pre-built vector database
├── aviation_documents.json   # Document metadata
└── README.md                 # This file
```

## Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google Gemini API key (required)

### Customization
- **Adjust sources**: Modify the `top_k` parameter in the sidebar (1-8 sources)
- **Enable debug**: Turn on debug mode to see processing details
- **Streaming control**: Toggle response streaming on/off

## Performance

- **Fast Loading**: Vector database loads in ~5 seconds
- **Quick Responses**: Typical response time 2-4 seconds
- **Efficient Search**: Semantic search across 123 document chunks
- **Source Attribution**: Relevance scores 0.0-1.0 for transparency

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your `.env` file contains a valid Google Gemini API key
2. **Port Conflicts**: Change port with `streamlit run streaming_app.py --server.port 8503`
3. **Dependencies**: Run `pip install -r requirements.txt` to install all required packages
4. **Database Issues**: Use `python manage_db.py status` to check vector database

### Database Management
```bash
# Check database status
python manage_db.py status

# Force rebuild (if needed)
python manage_db.py rebuild
```

## License

This project is for educational and demonstration purposes. Aviation content is sourced from public domain materials with proper attribution.

## Support

For technical issues or questions about aviation content, refer to the built-in help or check the source documents in the `aviation_data/` directory.
