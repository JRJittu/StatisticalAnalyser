import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY1")

class QueryAgent:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.combined_result_text = f.read()

        self.texts = self.split_text(self.combined_result_text, chunk_size=500, overlap=50)

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.embedding_model.encode(self.texts)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

        genai.configure(api_key=GOOGLE_API_KEY)
        self.llm_model = genai.GenerativeModel('gemini-2.0-flash')

    def split_text(self, text, chunk_size=500, overlap=50):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks
    
    def get_answer(self, query, k=3):
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k=k)

        retrieved_chunks = [self.texts[i] for i in I[0]]
        context = "\n\n".join(retrieved_chunks)
        print("contex: ", context)

        prompt = f"""Answer the question based on the following statistical analysis context:

        {self.combined_result_text}

        Question: {query}
        Answer:"""
        response = self.llm_model.generate_content(prompt)
        return response.text

# agent = QueryAgent("uploads/IRIS_result.txt")
# response = agent.get_answer("What is the mean of sepal length?")
# print("\nGemini Response:\n", response)
