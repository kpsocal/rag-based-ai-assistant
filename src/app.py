import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

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
    

    supported_extensions = {".txt", ".md"}

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext in supported_extensions and os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                results.append({
                    "content": content,
                    "metadata": {
                        "source": filename,
                        "title": os.path.splitext(filename)[0]
                    }
                })
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not results:
        print("No documents found in 'data/' directory. Using built-in samples.")
        # Add fallback samples if directory is empty
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
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
        llm_answer = ""
        # TODO: Implement the RAG query pipeline
        # HINT: Use self.vector_db.search() to retrieve relevant context chunks
        # HINT: Combine the retrieved document chunks into a single context string
        # HINT: Use self.chain.invoke() with context and question to generate the response
        # HINT: Return a string answer from the LLM

        # Your implementation here
        if not input.strip():
            return "Please ask a question."
        
        print(f"\nSearching for relevant context for: '{input}'")
        # Retrieve relevant chunks
        results = self.vector_db.search(input, n_results=n_results)

        # Combine retrieved documents into context
        context_pieces = results.get("documents", [])
        if not context_pieces:
            context = "No relevant information found in the knowledge base."
        else:
            context = "\n\n".join(context_pieces)

        # Invoke the chain with context and question
        llm_answer = self.chain.invoke({
            "context": context,
            "question": input
        })

        return llm_answer


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
                answer = assistant.invoke(question, n_results=4)
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
