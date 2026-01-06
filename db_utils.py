import os
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv

def get_vector_index():
    load_dotenv()

    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY is not set")

    # MODELS
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_base=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    Settings.llm = OpenAILike(
        model="openai.gpt-3.5-turbo",
        api_base=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ["OPENAI_API_KEY"],
        is_chat_model=True,
    )

    # --------------------------------------------------
    # CHROMA SETUP (PERSISTENT)
    # --------------------------------------------------
    chroma_client = chromadb.PersistentClient("./chroma_db")
    
    chroma_collection = chroma_client.get_or_create_collection(name="candidates")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # create index
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    
    return vector_index, chroma_collection
