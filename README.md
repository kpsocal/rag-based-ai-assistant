# RAG-Based AI Assistant - AAIDC Project 1 Template

## 🤖 What is this?

This repository introduces a Retrieval-Augmented Generation (RAG) based AI assistant designed to ingest PDF documents from company websites, such as annual 10-K filings submitted to the U.S. Securities and Exchange Commission (SEC) and provide answers to user queries. By leveraging vector databases, embedding models, and large language models (LLMs), the assistant enables efficient analysis of financial and operational data embedded in these reports.

**Think of it as:** ChatGPT that knows about Form 10K filing documents and can answer questions about them.

## 🎯 What does it do



- 📄 **Document Loading & Preprocessing 
- 🔍 **Chunking & Embedding
- 💬 **Retrieval
- 🧠 **Generation & Citation



## 📝 Implementation Steps

The project requires implementing 7 main steps:

1. **Prepare Your Documents** - Add your own documents to the data directory - Only PDFs with the year in the filename. Example - Fiscal2023Form10KAlphabet.pdf
2. **Document Loading** - Load documents from files into the system
3. **Text Chunking** - Split documents into smaller, searchable chunks
4. **Document Ingestion** - Process and store documents in the vector database  
5. **Similarity Search** - Find relevant documents based on queries
6. **RAG Prompt Template** - Design effective prompts for the LLM
7. **RAG Query Pipeline** - Complete query-response pipeline using retrieved context

---



**The RAG pipeline:**

1. Search for relevant chunks
2. Combine chunks into context
3. Generate response using LLM + context
4. Return structured results


---

## 🧪 Testing Your Implementation

### Test Individual Components

1. **Test chunking:**

   ```python
   from src.vectordb import VectorDB
   vdb = VectorDB()
   chunks = vdb.chunk_text("Your test text here...")
   print(f"Created {len(chunks)} chunks")
   ```
2. **Test document loading:**

   ```python
   documents = [{"content": "Test document", "metadata": {"title": "Test"}}]
   vdb.add_documents(documents)
   ```
3. **Test search:**

   ```python
   results = vdb.search("your test query")
   print(f"Found {len(results['documents'])} results")
   ```

### Test Full System

Once implemented, run:

```bash
python src/app.py
```

Try these example questions:

- "What is [topic from your documents]?"
- "Explain [concept from your documents]"
- "How does [process from your documents] work?"

---



## 🚀 Setup Instructions

### Prerequisites

Before starting, make sure you have:

- Python 3.8 or higher installed
- An API key from **one** of these providers:
  - [OpenAI](https://platform.openai.com/api-keys) (most popular)
  - [Groq](https://console.groq.com/keys) (free tier available)
  - [Google AI](https://aistudio.google.com/app/apikey) (competitive pricing)

### Quick Setup

1. **Clone and install dependencies:**

   ```bash
   git clone [your-repo-url]
   cd rt-aaidc-project1-template
   pip install -r requirements.txt
   ```

2. **Configure your API key:**

   ```bash
   # Create environment file (choose the method that works on your system)
   cp .env.example .env    # Linux/Mac
   copy .env.example .env  # Windows
   ```

   Edit `.env` and add your API key:

   ```
   OPENAI_API_KEY=your_key_here
   # OR
   GROQ_API_KEY=your_key_here  
   # OR
   GOOGLE_API_KEY=your_key_here
   ```


---

## 📁 Project Structure

```
rt-aaidc-project1-template/
├── src/
│   ├── app.py           # Main RAG application (implement Steps 2, 6-7)
│   └── vectordb.py      # Vector database wrapper (implement Steps 3-5)
├── data/               # Replace with your documents (Step 1)
│   ├── *.txt          # Your text files here
├── requirements.txt    # All dependencies included
├── .env.example       # Environment template
└── README.md          # This guide
```

---

## 🎓 Learning Objectives

By completing this project, you will:

- ✅ Understand RAG architecture and data flow
- ✅ Implement text chunking strategies
- ✅ Work with vector databases and embeddings
- ✅ Build LLM-powered applications with LangChain
- ✅ Handle multiple API providers
- ✅ Create production-ready AI applications

---

## 🏁 Success Criteria

Your implementation is complete when:

1. ✅ You can load your own documents
2. ✅ The system chunks and embeds documents
3. ✅ Search returns relevant results
4. ✅ The RAG system generates contextual answers
5. ✅ You can ask questions and get meaningful responses

**Good luck building your RAG system! 🚀**
