from pinecone import Pinecone


def initializing_pinecone(
    api_key: str,
    index_name: str,
    embeddings: str,
):
    """Initialize Pinecone with the given API key and environment."""
    pinecone = Pinecone(api_key=api_key)
    index = pinecone.Index(index_name)
    return pinecone, index


def uploading_data_to_pinecone(
    index,
    chunks,
    embeddings,
):
    chunks_texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = embeddings.embed_documents(chunks_texts)

    records = []
    for idx, (text, vector) in enumerate(zip(chunks_texts, chunk_embeddings)):
        records.append(
            {
                "id": str(idx),
                "values": vector,
                "metadata": {"text": text},
            }
        )

    index.upsert(vectors=records)
