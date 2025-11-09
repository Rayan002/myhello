"""

Indexing Agent - simplified RAG utilities.

Uses JsonLoader, a simple text splitter, and Chroma with a single persist_directory ('invoice.db').

"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any


from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain.vectorstores import Chroma # type: ignore
from langchain_core.documents import Document # type: ignore
import warnings



warnings.filterwarnings("ignore")


load_dotenv()

# Absolute DB directory at project root: invoice-auditor/invoice.db
DB_DIR = Path(__file__).resolve().parents[2] / "invoice.db"


def _load_erp_chunks() -> List[Document]:
 """
 Load and chunk ERP data files.
 Returns list of Document chunks ready for indexing.
 """
 data_dir = Path(__file__).resolve().parents[2] / "data" / "erp_mock_data"
 print("data -->", data_dir)
 json_files = [
     data_dir / "vendors.json",
     data_dir / "PO Records.json",
     data_dir / "sku_master.json",
 ]

 docs: List[Document] = []
 for jf in json_files:
     if jf.exists():
         # Load raw objects; we'll stringify dict contents for vector store compatibility
         loader = JSONLoader(str(jf), jq_schema='.[]', text_content=False)
         file_docs = loader.load()
         # Attach accurate metadata to every loaded doc
         for d in file_docs:
             # Ensure page_content is a string for Chroma
             if isinstance(d.page_content, (dict, list)):
                 try:
                     d.page_content = json.dumps(d.page_content, ensure_ascii=False)
                 except Exception:
                     d.page_content = str(d.page_content)
             if not d.metadata:
                 d.metadata = {}
             d.metadata.update({"is_accurate": "true", "source_file": jf.name})
         docs.extend(file_docs)

 if not docs:
     return []

 splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=20)
 chunks = splitter.split_documents(docs)
 return chunks



def add_chunks_to_db(chunks: List[Document]) -> Dict[str, Any]:
 """
 Add new chunks to the database.
 If DB doesn't exist, it will be created with ERP data + new chunks in a single operation.
 """
 if not chunks:
     return {"indexed": False, "chunks_count": 0, "error": "No chunks provided"}
 
 if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
     return {"indexed": False, "chunks_count": 0, "error": "HUGGINGFACEHUB_API_TOKEN is required"}
 
 from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
 embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
 
 try:
     
     # Check if DB needs to be initialized
     if not DB_DIR.exists():
        #  print(1)
         # Load ERP chunks and combine with new chunks in single operation
         erp_chunks = _load_erp_chunks()
         if not erp_chunks:
             return {
                 "indexed": False,
                 "chunks_count": 0,
                 "error": "Failed to load ERP data"
             }

         # Combine ERP chunks + new chunks and create DB in one call
         all_chunks = erp_chunks + chunks
         Chroma.from_documents(all_chunks, embeddings, persist_directory=str(DB_DIR))
         
         return {
             "indexed": True,
             "chunks_count": len(chunks),
             "erp_chunks_count": len(erp_chunks),
             "total_chunks": len(all_chunks)
         }
     else:
         # DB exists, just add new chunks
         db = Chroma(embedding_function=embeddings, persist_directory=str(DB_DIR))
         db.add_documents(chunks)
         
         return {"indexed": True, "chunks_count": len(chunks)}
         
 except Exception as e:
     return {"indexed": False, "chunks_count": 0, "error": str(e)}


def get_retriever_with_metadata(k: int = 3):
  """Get retriever with proper metadata filtering."""
  if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
      raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is required for retrieval")
  
  from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
  embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

  if not DB_DIR.exists():
      erp_chunks = _load_erp_chunks()
      if not erp_chunks:
          raise RuntimeError("Failed to load ERP data")
      Chroma.from_documents(erp_chunks, embeddings, persist_directory=str(DB_DIR))

  db = Chroma(embedding_function=embeddings, persist_directory=str(DB_DIR))

  # Chroma-native metadata filtering with $and, $eq, $in, $ne operators
  search_kwargs = {"k": k}

  # This returns a retriever that includes metadata
  return db.as_retriever(search_kwargs=search_kwargs)



__all__ = [

 "add_chunks_to_db",

 "get_retriever_with_metadata",

]
