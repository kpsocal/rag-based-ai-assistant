import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def get_content(doc:str) -> str:        
        return doc
    
    def get_metadata(doc_index:int) -> str:        
        return {"source": f"doc_{doc_index}"}

   

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    


    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # TODO: Implement text chunking logic
        # You have several options for chunking text - choose one or experiment with multiple:
        #
        # OPTION 1: Simple word-based splitting
        #   - Split text by spaces and group words into chunks of ~chunk_size characters
        #   - Keep track of current chunk length and start new chunks when needed
        #
        # OPTION 2: Use LangChain's RecursiveCharacterTextSplitter
        #   - from langchain_text_splitters import RecursiveCharacterTextSplitter
        #   - Automatically handles sentence boundaries and preserves context better
        #
        # OPTION 3: Semantic splitting (advanced)
        #   - Split by sentences using nltk or spacy
        #   - Group semantically related sentences together
        #   - Consider paragraph boundaries and document structure
        #
        # Feel free to try different approaches and see what works best!

        if not text.strip():
            return []

        chunks = []

        # Your implementation here  - Option 1
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space

            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                
                # Overlap: keep some words for next chunk
                current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                current_length = sum(len(w) + 1 for w in current_chunk)

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))    

        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        # TODO: Implement document ingestion logic
        # HINT: Loop through each document in the documents list
        # HINT: Extract 'content' and 'metadata' from each document dict
        # HINT: Use self.chunk_text() to split each document into chunks
        # HINT: Create unique IDs for each chunk (e.g., "doc_0_chunk_0")
        # HINT: Use self.embedding_model.encode() to create embeddings for all chunks
        # HINT: Store the embeddings, documents, metadata, and IDs in your vector database
        # HINT: Print progress messages to inform the user


        documentfordb = []
        metadata = []        
        ids = []
        all_embeddings = []


        print(f"Processing {len(documents)} documents...")
        # Your implementation here

        currentDocNumber = 0

        for index, doc in enumerate(documents, start=1):
            print(f"Processing Document number {index} started")
            if isinstance(doc, str):
                content = doc
                source_metadata = {"source": f"doc_{index}"}
            else:
                content = doc.get("content", str(doc))
                source_metadata = doc.get("metadata", {"source": f"doc_{index}"})
            
            contenttype = doc.get("metadata").get("filetype");
            #print(contenttype);
            currentDocNumber = index             
            
            chunks = self.chunk_text(content,2000)   
            
            if not chunks:
                print(f"Warning: Document {currentDocNumber} produced no chunks. Skipping.")
                continue
            
            # Generate embeddings for all chunks at once (faster)
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
            embeddings = embeddings.tolist()  # Chroma expects list of lists

            for indexChunk, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                print(f"Processing chunk number {indexChunk} in Document number {index} started")
                #print(f"Chunk = {chunk}") #for debugging only
                unique_id = f"doc_{index}_chunk_{indexChunk}"                
                documentfordb.append(chunk)
                ids.append(unique_id)
                metadata.append({**source_metadata, "doc_number": index, "chunk_number": indexChunk})                     
                all_embeddings.append(embedding)         

        # Add everything to ChromaDB in one batch call (efficient!)
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=all_embeddings,
                documents=documentfordb,
                metadatas=metadata
            )
            print(f"Successfully added {len(ids)} chunks to the vector database.")
        else:
            print("No chunks to add.")
        
        print("Documents added to vector database")
    

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # TODO: Implement similarity search logic
        # HINT: Use self.embedding_model.encode([query]) to create query embedding
        # HINT: Convert the embedding to appropriate format for your vector database
        # HINT: Use your vector database's search/query method with the query embedding and n_results
        # HINT: Return a dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        # HINT: Handle the case where results might be empty

        # Your implementation here

        if not query.strip():
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }
        
        query_embedding = self.embedding_model.encode([query])  # Returns a 2D array: (1, embedding_dim)
        query_embedding_list = query_embedding.tolist()[0]     # Convert to list of floats (1D)

        #Perform similarity search in ChromaDB
        results = self.collection.query(
        query_embeddings=[query_embedding_list],  # Chroma expects list of embeddings
        n_results=n_results,
        include=["documents", "metadatas", "distances"]  # data to return
        )

        if not results["ids"] or not results["ids"][0]:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }
        
        # return only first result
        return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
        "ids": results["ids"][0] if results["ids"] else [],
        }

        
