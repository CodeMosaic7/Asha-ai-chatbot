from langchain_text_splitters import CharacterTextSplitter


def split_text(text, chunk_size=1000, chunk_overlap=200):
    # If the input is a list, join it into a string

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )

    chunks = splitter.split_documents(text)
    return chunks
