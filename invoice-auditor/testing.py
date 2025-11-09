from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(embedding_function=embeddings, persist_directory=str(DB_DIR))

# Get a few docs directly from the vector store
docs = db.get()  # Returns all stored docs with metadata
print("Sample doc with metadata:")
print(docs)

# Check if metadata exists
if docs and docs['metadatas']:
    print(f"Metadata keys: {docs['metadatas'][0].keys()}")