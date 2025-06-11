

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


df = pd.read_csv(r'C:\Users\user\Documents\Python Scripts\BRAC\ecg_chatbot_project\data\discharge_ecg_text.csv')


model = SentenceTransformer('all-MiniLM-L6-v2')


ecg_embeddings = model.encode(df['ecg_text'].tolist(), show_progress_bar=True)


dimension = ecg_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(ecg_embeddings))


faiss.write_index(index, 'embeddings/ecg_index.faiss')

print("✅ FAISS index built and saved to embeddings/ecg_index.faiss")
