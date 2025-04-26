from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()


def loading_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load embeddings from HuggingFace."""
    login(os.getenv("HF_TOKEN"))
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
