from langchain_community.document_loaders import UnstructuredURLLoader


def loading_data(urls):
    """Load data from a list of URLs."""
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data
