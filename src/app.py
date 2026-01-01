import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import fitz 
import re

# Load environment variables
load_dotenv()


def load_documents() -> List[str]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """
    results = []
    # TODO: Implement document loading
    # HINT: Read the documents from the data directory
    # HINT: Return a list of documents
    # HINT: Your implementation depends on the type of documents you are using (.txt, .pdf, etc.)

    # Your implementation here

    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Warning: '{data_dir}' directory not found. Using sample documents instead.")
        # Fallback sample documents
        sample_texts = [
            "Python is a high-level, interpreted programming language known for its simplicity and readability.",
            "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation to improve LLM responses.",
            "ChromaDB is an open-source vector database designed for storing and querying embeddings efficiently.",
            "Sentence Transformers provide easy-to-use models for generating high-quality text embeddings."
        ]
        for i, text in enumerate(sample_texts, start=1):
            results.append({
                "content": text,
                "metadata": {"source": f"sample_doc_{i}", "title": f"Sample Document {i}"}
            })
        return results
    

    generic_supported_extensions = {".txt", ".md"}
    pdf_extensions = {".pdf"}

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        print(f"Processing Document: {filename}")
        year_match = re.search(r'(202[0-9])', filename)  # Finds 2024, 2025, etc.
        year = year_match.group(1) if year_match else "Unknown"

        if ext in generic_supported_extensions and os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                results.append({
                    "content": content,
                    "metadata": {
                        "source": filename,
                        "title": os.path.splitext(filename)[0],
                        "filetype":ext[1:],
                        "year": int(year)
                    }
                })
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        elif ext in pdf_extensions:
            try:
                print(f"Loading PDF: {filename}")
                with fitz.open(file_path) as doc:  # Use context manager — auto-closes safely
                    page_count = len(doc)

                    text_pages = []
                    for page in doc:
                        text = page.get_text("text")
                        text_pages.append(text)

                    full_text = "\n\n".join(text_pages)

                # Document auto-closed here — no manual close needed

                print(f"Successfully loaded {filename} ({page_count} pages, ~{len(full_text)//1000}k characters)")

                if full_text.strip():
                    results.append({
                        "content": full_text,
                        "metadata": {
                            "source": filename,
                            "title": os.path.splitext(filename)[0],
                            "filetype": "pdf",
                            "page_count": page_count,
                            "year": int(year)
                        }
                    })
                    print(f"Added {filename} to knowledge base")
                else:
                    print(f"Warning: No readable text found in {filename}")

            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")        


    if not results:
        print("No documents found in 'data/' directory. Using built-in samples.")        
        results = []
    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template
        # TODO: Implement your RAG prompt template
        # HINT: Use ChatPromptTemplate.from_template() with a template string
        # HINT: Your template should include placeholders for {context} and {question}
        # HINT: Design your prompt to effectively use retrieved context to answer questions
        self.prompt_template = None  # Your implementation here


        template = """
You are a helpful assistant that answers questions based only on the provided context.

Use the following retrieved context to answer the question. 
If you don't know the answer or it's not in the context, say "I don't know based on the provided information."

Context:
{context}

Question: {question}

Answer:
"""
        self.prompt_template = ChatPromptTemplate.from_template(template)


        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        if not documents:
            print("No documents to add.")
            return
        print(f"Adding {len(documents)} documents to the vector database...")
        self.vector_db.add_documents(documents)
        print("Documents added successfully.")

    def invoke(self, input: str, n_results: int = 3) -> str:
        if not input.strip():
            return "Please ask a question."

        print(f"\nSearching for relevant context for: '{input}'")
        
        # Retrieve relevant chunks
        results = self.vector_db.search(input, n_results=n_results)

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        # for debugging only
        #print(f"Retrieved {len(documents)} relevant chunks")
        #for i, doc in enumerate(documents[:3]):  # Print first 3
        #     print(f"Chunk {i+1}: {doc[:200]}...")  # First 200 chars

        # Build context
        if not documents:
            context = "No relevant information found in the knowledge base."
            final_answer = self.chain.invoke({
                "context": context,
                "question": input
            })
            return final_answer  # → "I don't know..." with NO sources

        else:
            # There ARE relevant documents
            context_pieces = []
            for doc, meta in zip(documents, metadatas):
                year = meta.get("year", "Unknown")
                source = meta.get("source", "Unknown")
                chunk_num = meta.get("chunk_number", "?")
                prefix = f"[From fiscal {year} 10-K, {source}, Chunk {chunk_num}]\n"
                context_pieces.append(prefix + doc)
            
            context = "\n\n".join(context_pieces)     

            # Build sources ONLY when we have real context
            sources = []
            seen = set()
            for meta in metadatas:
                source_file = meta.get("source", "Unknown file")
                page = meta.get("page_number")
                chunk = meta.get("chunk_number")

                if page is not None:
                    citation = f"{source_file} (Page {page})"
                else:
                    citation = f"{source_file} (Chunk {chunk + 1 if chunk is not None else '?'})"

                if citation not in seen:
                    seen.add(citation)
                    sources.append(citation)

            source_note = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)

            # Generate answer
            llm_answer = self.chain.invoke({
                "context": context,
                "question": input
            })

            # Append sources only when we used real context
            return llm_answer + source_note


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        print("\n" + "="*50)
        print("RAG Assistant is ready! Ask questions or type 'quit' to exit.")
        print("="*50)

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                answer = assistant.invoke(question, n_results=6)
                print(f"\nAssistant: {answer}")

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")
        print("  • A 'data/' folder with .txt or .md files (or it will use samples)")


if __name__ == "__main__":
    main()
