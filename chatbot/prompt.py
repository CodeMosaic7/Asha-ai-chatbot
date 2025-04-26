from langchain.prompts import PromptTemplate


def build_prompt_template():
    template = """
You are a helpful assistant who gives information about job opportunities specifically targeted towards women.

Use the following retrieved context to answer the user's query.
If you don't know the answer, say "I couldn't find a suitable opportunity at the moment."

Context:
{context}

Question:
{question}

Helpful Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    return prompt
