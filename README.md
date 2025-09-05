# Aviation RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for aviation knowledge using Google's Gemini API and Streamlit.

## 🚀 Features

- **Smart Retrieval**: Uses Gemini embeddings to find relevant aviation documents
- **Expert Responses**: Generates accurate answers using Gemini AI with retrieved context
- **Source Attribution**: Shows which documents were used to generate each answer
- **Clean UI**: Simple Streamlit interface for easy interaction
- **Aviation Focus**: Specialized knowledge base covering flight operations, safety, regulations, and technical concepts

## 📋 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Gemini API
1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add your API key to the `.env` file:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

### 3. Run the Application
```bash
streamlit run app.py
```

## 📁 Project Structure

```
airline_rag/
├── aviation_data/           # Aviation knowledge base
│   ├── wikipedia/          # Wikipedia articles
│   ├── faa/               # FAA manuals
│   ├── eurocontrol/       # EUROCONTROL guidelines
│   └── open_data/         # Reference data
├── app.py                 # Streamlit frontend
├── rag_system.py         # RAG backend logic
├── requirements.txt      # Python dependencies
└── .env                  # API configuration
```

## 🎯 How It Works

1. **Document Loading**: Loads aviation text files and splits them into chunks
2. **Embedding Creation**: Uses Gemini embedding model to create vector representations
3. **Vector Search**: FAISS index enables fast similarity search
4. **Context Retrieval**: Finds most relevant documents for each question
5. **Response Generation**: Gemini generates answers using retrieved context

## 💡 Usage Examples

Try asking questions like:
- "What is ETOPS and why is it important?"
- "How does flight planning work for commercial aircraft?"
- "What are IATA delay codes used for?"
- "Explain the role of a flight dispatcher"

## 🔧 Configuration Options

### Environment Variables (.env)
- `GOOGLE_API_KEY`: Your Gemini API key (required)
- `GEMINI_TEMPERATURE`: Response creativity (0.0-1.0, default: 0.7)
- `GEMINI_MODEL`: Model version (default: gemini-1.5-flash)
- `EMBEDDING_MODEL`: Embedding model (default: models/embedding-001)

### Streamlit Settings
- Adjust number of source documents (1-10)
- Toggle source document display
- View system status in sidebar

## 📊 Knowledge Base Coverage

The system includes aviation content covering:
- **Flight Operations**: Planning, dispatching, procedures
- **Safety Systems**: ETOPS, MEL, safety protocols
- **Technical Concepts**: Aerodynamics, fuel efficiency, performance
- **Regulatory Framework**: FAA procedures, international standards  
- **Airport Operations**: Collaborative decision making, efficiency

## 🛡️ License & Attribution

All content maintains proper attribution:
- **Wikipedia**: Creative Commons Attribution-ShareAlike
- **FAA**: Public Domain (US Government Works)
- **EUROCONTROL**: Creative Commons Attribution

## 🚨 Troubleshooting

### Common Issues:
1. **API Key Error**: Make sure your Gemini API key is correctly set in `.env`
2. **Import Error**: Install all requirements with `pip install -r requirements.txt`
3. **Slow Performance**: Reduce the number of source documents in settings
4. **Memory Issues**: The system loads all embeddings in memory; reduce chunk size if needed

### Getting Help:
- Check the Streamlit sidebar for system status
- Review the console output for detailed error messages
- Ensure your API key has sufficient quota

## 🔮 Future Enhancements

- Persistent vector database (Pinecone, Weaviate)
- Multiple file upload support
- Advanced filtering and search options
- Conversation memory and follow-up questions
- Export functionality for Q&A pairs
