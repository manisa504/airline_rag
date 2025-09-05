# SETUP INSTRUCTIONS FOR MANISHA

## Quick Deployment to GitHub

### Step 1: Create GitHub Repository
1. Go to github.com and log in to your account
2. Click "New repository" (green button)
3. Repository name: `aviation-knowledge-assistant` 
4. Description: `AI-powered aviation chatbot with RAG technology`
5. Make it **Public** 
6. Check "Add a README file"
7. Click "Create repository"

### Step 2: Upload Files
1. In your new repository, click "uploading an existing file"
2. Drag and drop ALL files from this folder EXCEPT:
   - This SETUP_INSTRUCTIONS.md file
   - Any .DS_Store files
   - Any __pycache__ folders

### Step 3: Get Google Gemini API Key
1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API key"
3. Copy the key (starts with "AIza...")
4. In your GitHub repo, create a file called `.env`
5. Add this line: `GOOGLE_API_KEY=your_api_key_here`

### Step 4: Test Locally (Optional)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streaming_app.py
```

### Step 5: Deploy Online (Optional)
You can deploy this to:
- **Streamlit Cloud**: Connect your GitHub repo at share.streamlit.io
- **Heroku**: Add `Procfile` and deploy
- **Railway**: Simple deployment from GitHub

## What This Project Does

This is an **AI Aviation Expert** that can answer questions about:
- Flight operations and procedures
- Safety regulations and protocols  
- Aircraft technical specifications
- Airport operations and management
- Aviation delay codes and terminology

The AI provides **source attribution** - showing exactly which aviation documents it used to answer your questions.

## Sample Questions to Try
- "What is ETOPS?"
- "How does flight planning work?"
- "What are IATA delay codes?"
- "What is a flight dispatcher's role?"

## Technical Features
- ✅ Real-time streaming responses
- ✅ Source attribution with relevance scores
- ✅ 123+ aviation document knowledge base
- ✅ Clean, professional interface
- ✅ Fast vector search technology

---

**Your aviation chatbot is ready to go! 🛩️**
