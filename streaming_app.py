import streamlit as st
import os
from simple_rag import SimpleAviationRAG
import time
from typing import Dict, List, Generator

# Page configuration
st.set_page_config(
    page_title="Aviation Knowledge Assistant",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS for clean UI
st.markdown("""
<style>
    /* Compact sidebar */
    .css-1d391kg {
        width: 250px !important;
    }
    .css-1lcbmhc {
        width: 250px !important;
    }
    
    /* Main header */
    .main-header {
        font-size: 1.8rem;
        color: #1E40AF;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Compact sidebar elements */
    .sidebar-section {
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.3rem;
    }
    
    /* Status indicators */
    .status-indicator {
        padding: 0.3rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 0.3rem 0;
    }
    .status-success {
        background-color: #10B981;
        color: white;
    }
    .status-processing {
        background-color: #F59E0B;
        color: white;
    }
    
    /* Compact source display */
    .source-mini {
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        font-size: 0.8rem;
        border-left: 3px solid #3B82F6;
    }
    
    .source-header {
        font-weight: 600;
        color: #475569;
        font-size: 0.75rem;
        margin-bottom: 0.2rem;
    }
    
    .source-content {
        color: #64748B;
        font-size: 0.7rem;
        line-height: 1.3;
    }
    
    /* Question list styling */
    .question-item {
        font-size: 0.8rem;
        padding: 0.3rem 0;
        color: #4B5563;
        cursor: pointer;
        border-bottom: 1px solid #E5E7EB;
    }
    
    .question-item:hover {
        color: #1E40AF;
        background-color: #F8FAFC;
    }
    
    /* Compact metrics */
    .metric-mini {
        background: #4F46E5;
        color: white;
        padding: 0.4rem;
        border-radius: 4px;
        text-align: center;
        margin: 0.2rem 0;
        font-size: 0.75rem;
    }
    
    /* Smaller chat elements */
    .stChatMessage {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def stream_text(text: str, delay: float = 0.02) -> Generator[str, None, None]:
    """Stream text word by word for better UX."""
    words = text.split()
    current_text = ""
    
    for word in words:
        current_text += word + " "
        yield current_text
        time.sleep(delay)

@st.cache_resource(show_spinner="🔄 Initializing Aviation Knowledge Base...")
def initialize_rag():
    """Initialize the RAG system with loading indicator."""
    try:
        return SimpleAviationRAG()
    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")
        return None

def display_sources_compact(sources: List[Dict]):
    """Compact source display in a collapsible dropdown."""
    if not sources:
        return
    
    # Quick summary line
    avg_score = sum(s.get('similarity_score', 0) for s in sources) / len(sources)
    unique_sources = len(set(s.get('source', 'unknown') for s in sources))
    
    # Collapsible sources section
    with st.expander(f"📚 View Sources ({len(sources)} sources, avg score: {avg_score:.2f})", expanded=False):
        # Create a container with scrollable sources
        sources_container = st.container()
        
        with sources_container:
            # Use columns for better layout
            for i, source in enumerate(sources):
                relevance = "🟢" if source.get('similarity_score', 0) > 0.7 else "🟡" if source.get('similarity_score', 0) > 0.5 else "🔴"
                
                # Use markdown with proper escaping
                st.markdown(f"""
                **{relevance} Source {i+1}: {source.get('source', 'Unknown')}** *(Relevance: {source.get('similarity_score', 0):.3f})*
                
                {source.get('content', '')[:400]}{'...' if len(source.get('content', '')) > 400 else ''}
                
                ---
                """)
        
        # Add CSS to make the expander content scrollable
        st.markdown("""
        <style>
        .streamlit-expanderContent {
            max-height: 400px;
            overflow-y: auto;
        }
        </style>
        """, unsafe_allow_html=True)

def process_user_question(question: str, rag, num_sources: int, streaming_enabled: bool, show_debug: bool):
    """Process a user question and return the response."""
    # Show processing status
    if show_debug:
        status_placeholder = st.empty()
        status_placeholder.markdown('<div class="status-indicator status-processing">🔍 Searching knowledge base...</div>', unsafe_allow_html=True)
    
    try:
        # Query the RAG system
        with st.spinner("🔍 Analyzing your question..."):
            result = rag.query(question, top_k=num_sources)
        
        if show_debug:
            status_placeholder.markdown('<div class="status-indicator status-processing">🤖 Generating response...</div>', unsafe_allow_html=True)
        
        # Display answer with streaming if enabled
        if streaming_enabled and result['answer']:
            response_placeholder = st.empty()
            
            # Stream the response
            for partial_text in stream_text(result['answer'], delay=0.03):
                response_placeholder.markdown(partial_text)
            
            if show_debug:
                status_placeholder.markdown('<div class="status-indicator status-success">✅ Response generated successfully!</div>', unsafe_allow_html=True)
                time.sleep(1)
                status_placeholder.empty()
        else:
            # Show full response immediately
            st.markdown(result['answer'])
        
        # Show sources
        if result['sources']:
            display_sources_compact(result['sources'])
        
        return result
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error while processing your question: {str(e)}"
        st.error(error_msg)
        
        if show_debug:
            st.code(f"Debug info: {str(e)}")
        
        return {
            'question': question,
            'answer': error_msg,
            'sources': []
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">✈️ Aviation Knowledge Assistant</h1>', unsafe_allow_html=True)
    
    # App description
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 1rem; background-color: #F8FAFC; border-radius: 8px; border-left: 4px solid #3B82F6;">
        <p style="margin: 0; color: #475569; font-size: 0.95rem; line-height: 1.5;">
            🚁 <strong>AI-powered aviation expert</strong> trained on comprehensive flight operations, safety procedures, regulations, and technical concepts. 
            Get instant answers with <strong>source attribution</strong> from authoritative aviation documents including ETOPS, MEL, delay codes, and more.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history and state first
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "👋 **Welcome!** I'm ready to help with your aviation questions. Try asking about flight operations, safety procedures, regulations, or technical concepts. You can also use the sample questions in the sidebar to get started!",
            "sources": []
        })
    
    # Initialize process_question flag
    if "process_question" not in st.session_state:
        st.session_state.process_question = False
    
    # Initialize pending_question
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    
    # Handle pending question from sidebar
    if st.session_state.pending_question:
        st.session_state.messages.append({"role": "user", "content": st.session_state.pending_question})
        st.session_state.pending_question = None
    
    # Sidebar with compact info
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🛩️ System Status</div>', unsafe_allow_html=True)
        
        # API Status
        api_status = "✅" if os.getenv('GOOGLE_API_KEY') else "❌"
        st.markdown(f'<div class="sidebar-section">**API:** {api_status}</div>', unsafe_allow_html=True)
        
        # Initialize RAG and show status
        rag = initialize_rag()
        if rag:
            st.markdown('<div class="sidebar-section">**RAG:** ✅ Ready</div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section">**DB:** 123 docs</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sidebar-section">**RAG:** ❌ Failed</div>', unsafe_allow_html=True)
            st.error("Check configuration")
            return
        
        st.markdown("---")
        
        # Compact settings
        st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
        
        num_sources = st.slider(
            "Sources",
            min_value=1,
            max_value=8,
            value=3,
            help="Number of sources to use"
        )
        
        streaming_enabled = st.checkbox("Stream responses", value=True)
        show_debug = st.checkbox("Debug mode", value=False)
        
        st.markdown("---")
        
        # Sample questions as simple list
        st.markdown('<div class="sidebar-title">💡 Sample Questions</div>', unsafe_allow_html=True)
        
        questions = [
            "What is ETOPS?",
            "How does flight planning work?",
            "What are IATA delay codes?",
            "What is a flight dispatcher?",
            "What is MEL?",
            "What is A-CDM?",
            "How does Mach number affect flight?",
            "What affects fuel economy?"
        ]
        
        # Sample questions as clickable items
        for i, question in enumerate(questions):
            # Use a unique key for each button and handle the click
            if st.button(f"• {question}", key=f"q_{i}", use_container_width=True):
                # Use a session state flag to trigger processing
                st.session_state.pending_question = question
                st.rerun()
        
        st.markdown("---")
        
        # Quick actions
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.session_state.process_question = False
            st.session_state.pending_question = None
            st.rerun()
    
    # Chat interface
    st.markdown("### 💬 Aviation Knowledge Assistant")
    
    # Check if the last message is a user message that needs processing
    needs_processing = False
    if (len(st.session_state.messages) > 0 and 
        st.session_state.messages[-1]["role"] == "user" and
        (len(st.session_state.messages) == 1 or 
         st.session_state.messages[-2]["role"] == "assistant")):
        needs_processing = True
    
    # Display chat history (except the last user message if it needs processing)
    messages_to_display = st.session_state.messages[:-1] if needs_processing else st.session_state.messages
    
    for message in messages_to_display:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Show assistant message with better formatting
                st.markdown(message["content"])
                
                # Show sources if available
                if message.get("sources") and len(message["sources"]) > 0:
                    display_sources_compact(message["sources"])
            else:
                # User message
                st.markdown(message["content"])
    
    # Process the pending user message if needed
    if needs_processing:
        user_question = st.session_state.messages[-1]["content"]
        
        # Display the user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            result = process_user_question(user_question, rag, num_sources, streaming_enabled, show_debug)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result['answer'],
                "sources": result['sources']
            })
    
    # Chat input
    if prompt := st.chat_input("Ask your aviation question... ✈️"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            result = process_user_question(prompt, rag, num_sources, streaming_enabled, show_debug)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result['answer'],
                "sources": result['sources']
            })

if __name__ == "__main__":
    main()
