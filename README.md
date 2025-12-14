# BookChat-AI

An AI-powered book chat system that uses RAG (Retrieval-Augmented Generation) to provide intelligent conversations about books, literature, and reading.

## Features

- **Intelligent Book Discussions**: Chat about books, authors, genres, and literary topics
- **LLM-Based Query Validation**: Automatically determines if queries are book-related using AI
- **RAG-Powered Knowledge**: Automatically loads comprehensive literary knowledge from PDF documents
- **Smart Retrieval**: Uses vector similarity search to find relevant context for queries
- **Conversational Memory**: Maintains context across conversations
- **OpenLibrary Integration**: Access to millions of books and author information
- **AWS Bedrock Integration**: Uses Amazon's Nova and Titan models for embeddings and generation

## Quick Start

### Prerequisites

- Python 3.10+
- AWS Account with Bedrock access
- AWS credentials configured

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv bookchat-env
   source bookchat-env/bin/activate  # On Windows: bookchat-env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in `.env`:
   ```
   # AWS Credentials
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   
   # Model Configuration
   LLM_MODEL=amazon.nova-lite-v1:0
   EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
   LLM_TEMPERATURE=0.7
   LLM_MAX_TOKENS=5000
   
   # Optional: LangSmith Tracing (for debugging)
   LANGCHAIN_TRACING_V2=false
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=bookchat-ai
   ```

### Running the App

```bash
streamlit run app.py
```

## PDF Knowledge Base

The system automatically loads literary knowledge from `knowledge_base/literary_knowledge.pdf` on first launch:

- **Automatic Loading**: PDF is ingested into the RAG system when the app starts
- **Smart Caching**: Already-loaded PDFs are detected and not re-ingested
- **Persistent Storage**: Knowledge is stored in ChromaDB and persists across sessions
- **Rich Content**: Includes information about classic literature, authors, genres, and literary analysis

### Current Knowledge Base
- 185 total documents in the RAG system
- 142 chunks from literary_knowledge.pdf
- Covers classic literature, authors, genres, literary devices, and more

### Adding New Knowledge

To add new knowledge to the system:
1. Place your PDF file in the `knowledge_base/` directory
2. Update the `_load_pdf_knowledge()` method in `utils/book_rag.py` to include your PDF
3. Restart the app

To refresh the knowledge base:
```bash
# Clear the existing database and reload
rm -rf chroma_db/
# Restart the app - it will rebuild the knowledge base
```

## Architecture

- **Frontend**: Streamlit web interface with multi-turn conversation UI
- **Backend**: LangChain agent orchestration with intelligent routing
- **Query Validation**: LLM-based book-relevance classification
- **Vector Store**: ChromaDB with persistent storage (185 documents)
- **LLM**: AWS Bedrock Amazon Nova Lite v1
- **Embeddings**: AWS Bedrock Titan Embed Text v2
- **External API**: OpenLibrary API for real-time book metadata
- **Memory**: ConversationBufferWindowMemory (last 10 turns)
- **Agent Tools**: 3 specialized tools (search books, get book info, analyze themes)
- **Monitoring**: LangSmith tracing support

## Project Structure

```
BookChat-AI/
├── app.py                          # Main Streamlit application
├── book_chat_system.py             # Core chat system orchestrator
├── .env                            # Environment variables (create from example below)
├── requirements.txt                # Python dependencies
├── knowledge_base/                 # PDF knowledge files
│   └── literary_knowledge.pdf      # 142 chunks of literary knowledge
├── utils/                          # Core modules
│   ├── __init__.py
│   ├── book_rag.py                # RAG system with ChromaDB integration
│   ├── book_tools.py              # LangChain tools for agent
│   ├── openlibrary_api.py         # OpenLibrary API client
│   └── prompt_templates.py        # System prompts and templates
├── chroma_db/                      # Vector database 
```

## License

MIT License