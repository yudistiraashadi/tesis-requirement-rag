import json
import os
from typing import List

import numpy as np
from openai import OpenAI
from PyPDF2 import PdfReader

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes

# Set OpenAI API key
client = OpenAI()


def get_pdf_text(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def split_text(text: str, chunk_size: int = 1000) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model="text-embedding-3-small")
        .data[0]
        .embedding
    )


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_most_relevant_chunks(
    query: str, chunks: List[str], embeddings: List[List[float]], top_k: int = 3
):
    query_embedding = get_embedding(query)
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    top_indices = np.argsort(similarities)[-top_k:]
    return [chunks[i] for i in top_indices]


def get_chat_response(
    query: str, context: str, conversation_history: List[dict]
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions about thesis requirements based on the provided context.",
        },
    ]
    messages.extend(conversation_history)
    messages.append(
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    )

    print("Generating chat response...")
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, stream=True
    )

    # Stream the response
    full_response = ""
    print("\nAssistant: ", end="", flush=True)

    try:
        for chunk in response:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
    except Exception as e:
        print(f"\nError during streaming: {e}")

    print("\n")  # Add newline at the end
    return full_response


def save_embeddings(chunks: List[str], embeddings: List[List[float]], filename: str):
    data = {"chunks": chunks, "embeddings": embeddings}
    with open(filename, "w") as f:
        json.dump(data, f)


def load_embeddings(filename: str) -> tuple[List[str], List[List[float]]]:
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            return data["chunks"], data["embeddings"]
    return [], []


def main():
    print("Welcome to Thesis Requirements Chat CLI!")

    # Initialize state
    chunks = []
    embeddings = []
    conversation_history = []

    # Use default PDF
    default_pdf = "./Panduan Penyusunan Tesis 2021.pdf"
    embeddings_file = "panduan_tesis_embeddings.json"

    if not os.path.exists(default_pdf):
        print("Error: Default PDF not found")
        return

    # Process embeddings
    print("Loading embeddings...")
    chunks, embeddings = load_embeddings(embeddings_file)
    if not chunks:
        print("Generating new embeddings...")
        text = get_pdf_text(default_pdf)
        chunks = split_text(text)
        embeddings = [get_embedding(chunk) for chunk in chunks]
        save_embeddings(chunks, embeddings, embeddings_file)
    print("Embeddings loaded successfully!")

    # Main chat loop
    print("\nYou can start asking questions about thesis requirements.")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    while True:
        user_question = input("\nYour question: ").strip()

        if user_question.lower() in ["quit", "exit"]:
            print("Thank you for using Thesis Requirements Chat!")
            break

        if not user_question:
            continue

        print("\nProcessing your question...")

        relevant_chunks = find_most_relevant_chunks(user_question, chunks, embeddings)
        context = "\n".join(relevant_chunks)

        # The response will be streamed in the get_chat_response function
        response = get_chat_response(user_question, context, conversation_history)

        # Update conversation history
        conversation_history.extend(
            [
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": response},
            ]
        )


if __name__ == "__main__":
    main()
