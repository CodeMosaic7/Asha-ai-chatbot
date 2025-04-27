---

# 🚀 ASHA - Your Helpful AI Assistant

**A Retrieval-Augmented Generation (RAG) Chatbot using GPT-2, LangChain, Pinecone, and Streamlit**

---

## 📜 Overview
**ASHA** is a Retrieval-Augmented Generation (RAG) based chatbot designed to give intelligent, contextual answers.  
It uses **LangChain** to manage retrieval and generation, **Pinecone** for storing and querying vectors, **GPT-2** for response generation, and a **Streamlit** app for the frontend.

---

## 🚀 Features
- 🔎 **Contextual retrieval** of information using **Pinecone** vector database.
- 🧠 **GPT-2** model generates human-like responses based on retrieved knowledge.
- 🔗 **LangChain** integration for flexible retrieval-generation workflows.
- 🖥️ **Streamlit UI** for a clean and interactive user experience.
- ⚡ Fast, intelligent, and memory-efficient design.

---

## 🧩 How it Works
1. **User inputs a query**.
2. **Pinecone** retrieves top relevant documents.
3. Retrieved context + query is formatted into a **custom prompt**.
4. **GPT-2** generates a coherent, helpful answer.
5. The **Streamlit** app displays the response.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **LangChain**
- **Pinecone**
- **Huggingface Transformers (GPT-2)**
- **Streamlit**
- **FAISS** (optional for local retrieval without Pinecone)

---

## 📂 Project Structure
```bash
.
├── app.py                # Main Streamlit application
├── chatbot/              
│   ├── embed_text.py      # Text embedding utilities
│   ├── load_data.py       # Load and prepare documents
│   ├── model.py           # Load GPT-2 model
│   ├── pinecone_setup.py  # Setup and manage Pinecone connection
│   ├── prompt.py          # Build custom prompt templates
│   ├── retrieval.py       # Retrieve context from Pinecone
│   ├── split_text.py      # Text splitting into chunks
│   ├── util.py            # Utility functions
├── requirements.txt       # Python dependencies
├── .gitignore             # Files/folders to ignore by Git
└── .devcontainer/         # Dev container setup for VS Code (optional)
```

---

## 🏗️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables
Create a `.env` file inside the `chatbot/` folder and add:
```ini
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
HUGGINGFACE_API_KEY=your-huggingface-api-key
```

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📚 Example Query

> **User:** "What are the job opportunities for women in India?"

> **Bot:**  
"There are several job opportunities for women in India, especially in companies like Google, Accenture, and IBM. Positions like Human Resource Specialist, Software Developer, and Healthcare Professional are quite common."

---

## ✨ Future Improvements
- Add **chat history** (memory feature).
- Enable **switching between multiple LLMs** (GPT-2, GPT-3, custom LLMs).
- Add **authentication** for secure document uploads.
- Improve **UI/UX** with chat animations and avatars.

---

## 📝 License
This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute it!

---

## 🙌 Acknowledgements
- [LangChain](https://www.langchain.com/)
- [Pinecone](https://www.pinecone.io/)
- [Huggingface GPT-2](https://huggingface.co/gpt2)
- [Streamlit](https://streamlit.io/)

---
