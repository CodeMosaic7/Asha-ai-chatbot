# import sys

# # Skip torch modules from Streamlit's file watcher
# original_modules = sys.modules.copy()
# for k in list(original_modules.keys()):
#     if k.startswith("torch"):
#         sys.modules.pop(k)

import streamlit as st
import dotenv
from chatbot.load_data import loading_data
from chatbot.split_text import split_text
from chatbot.embed_text import loading_embeddings
from chatbot.pinecone_setup import initializing_pinecone, uploading_data_to_pinecone
from chatbot.model import load_hf_model
from chatbot.util import download_nltk
from chatbot.retrieval import retrieve_context
from chatbot.prompt import build_prompt_template
from huggingface_hub import login
import os


def main():
    st.title("🔍 Women-Focused Jobs Assistant")
    st.write("This AI assistant helps find women-focused job opportunities.")

    dotenv.load_dotenv()

    download_nltk()

    URLs = [
        "https://www.herkey.com/jobs",
        "https://powertofly.com/jobs",
        "https://leanin.org/circles",
        "https://internshala.com/internships/work-from-home-internships/women/",
        "https://www.womenwhocode.com/jobs",
        "https://www.womentech.net/jobs",
    ]
    print("Data loaded from URLs: ", URLs)
    st.sidebar.title("Settings")
    hf_token = os.environ.get("HF_TOKEN")
    login(hf_token)

    embeddings = loading_embeddings()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index = os.getenv("PINECONE_INDEX")

    pc, index = initializing_pinecone(pinecone_api_key, pinecone_index, embeddings)

    with st.spinner("Loading website data..."):
        data = loading_data(URLs)
        chunks = split_text(data)
        uploading_data_to_pinecone(index, chunks, embeddings)
        st.success(f"Data loaded and {len(chunks)} chunks indexed successfully!")

    with st.spinner("Loading model (one-time)..."):
        llm = load_hf_model(token=hf_token)
        prompt_template = build_prompt_template()

    user_query = st.text_input("Ask a question related to jobs for women:")

    if user_query:
        with st.spinner("Retrieving context and generating answer..."):
            context = retrieve_context(user_query, index=index, embeddings=embeddings)
            full_prompt = prompt_template.format(context=context, question=user_query)
            response = llm.predict(full_prompt)

            st.success("Answer:")
            st.write(response)


if __name__ == "__main__":
    main()
