import os
import json
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

DB_DIR = Path(__file__).resolve().parents[2] / "invoice.db"


def _load_erp_chunks() -> List[Document]:
    data_dir = Path(__file__).resolve().parents[2] / "data" / "erp_mock_data"
    json_files = [
        data_dir / "vendors.json",
        data_dir / "PO Records.json",
        data_dir / "sku_master.json",
    ]

    docs = []
    for jf in json_files:
        if not jf.exists():
            continue
        
        loader = JSONLoader(str(jf), jq_schema='.[]', text_content=False)
        file_docs = loader.load()
        
        for d in file_docs:
            if isinstance(d.page_content, (dict, list)):
                d.page_content = json.dumps(d.page_content, ensure_ascii=False)
            
            if not d.metadata:
                d.metadata = {}
            d.metadata.update({"is_accurate": "true", "source_file": jf.name})
        
        docs.extend(file_docs)

    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=20)
    return splitter.split_documents(docs)


def add_chunks_to_db(chunks: List[Document]) -> Dict[str, Any]:
    if not chunks:
        return {"indexed": False, "chunks_count": 0, "error": "No chunks provided"}
    
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        return {"indexed": False, "chunks_count": 0, "error": "HUGGINGFACEHUB_API_TOKEN is required"}
    
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    try:
        if not DB_DIR.exists():
            erp_chunks = _load_erp_chunks()
            if not erp_chunks:
                return {"indexed": False, "chunks_count": 0, "error": "Failed to load ERP data"}
            
            all_chunks = erp_chunks + chunks
            Chroma.from_documents(all_chunks, embeddings, persist_directory=str(DB_DIR))
            return {
                "indexed": True,
                "chunks_count": len(chunks),
                "erp_chunks_count": len(erp_chunks),
                "total_chunks": len(all_chunks)
            }
        else:
            db = Chroma(embedding_function=embeddings, persist_directory=str(DB_DIR))
            db.add_documents(chunks)
            return {"indexed": True, "chunks_count": len(chunks)}
    except Exception as e:
        return {"indexed": False, "chunks_count": 0, "error": str(e)}


def get_retriever_with_metadata(k: int = 3):
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is required for retrieval")
    
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if not DB_DIR.exists():
        erp_chunks = _load_erp_chunks()
        if not erp_chunks:
            raise RuntimeError("Failed to load ERP data")
        Chroma.from_documents(erp_chunks, embeddings, persist_directory=str(DB_DIR))

    db = Chroma(embedding_function=embeddings, persist_directory=str(DB_DIR))
    return db.as_retriever(search_kwargs={"k": k})


__all__ = ["add_chunks_to_db", "get_retriever_with_metadata"]
