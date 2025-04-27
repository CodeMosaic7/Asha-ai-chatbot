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
from streamlit_lottie import st_lottie
import requests


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_chat = load_lottieurl(
    "https://assets7.lottiefiles.com/packages/lf20_qp1q7mct.json"
)

st_lottie(lottie_chat, height=300, key="chat")

st.markdown(
    """
    <style>
    .stApp {
        font-family: 'Courier New', monospace; /* Typewriter font */
    }
    .big-font {
        font-size: 36px !important;
        color: #4CAF50;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display header
st.markdown('<div class="big-font">Welcome!</div>', unsafe_allow_html=True)


def main():
    st.title("🔍 ASHA AI CHATBOT")
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
        "https://internshala.com/jobs-for-women/",
        "https://apna.co/jobs/female-jobs-in-lucknow",
    ]
    print("Data loaded from URLs: ", URLs)
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
            response = llm.invoke(full_prompt)

            st.success("Answer:")
            st.write(response)


if __name__ == "__main__":
    main()
