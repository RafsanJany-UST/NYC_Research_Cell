# run_gradio_chatbot.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import gradio as gr


df = pd.read_csv('data/discharge_ecg_text.csv')
index = faiss.read_index('embeddings/ecg_index.faiss')


embed_model = SentenceTransformer('all-MiniLM-L6-v2')


llm_pipeline = pipeline("text-generation", model="gpt2")

def chatbot_ecg(user_input):

    query_embedding = embed_model.encode([user_input])
    D, I = index.search(np.array(query_embedding), k=5)
    retrieved_chunks = [df['ecg_text'].iloc[i] for i in I[0]]

    context = "\n".join(retrieved_chunks)
    prompt = f"""
    You are an expert ECG assistant.

    Here is the context from patient discharge notes:

    {context}

    User question: {user_input}

    Answer based on the context and general ECG knowledge:
    """
    

    response = llm_pipeline(prompt, max_length=512, num_return_sequences=1)
    return response[0]['generated_text']


gr.Interface(fn=chatbot_ecg, inputs="text", outputs="text", title="ECG Assistant Chatbot").launch()
