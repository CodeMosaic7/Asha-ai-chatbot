from langchain_pinecone import PineconeVectorStore


def retrieve_context(query, index, embeddings, top_k=3):
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": 0.5},
    )
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context
