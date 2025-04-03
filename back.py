import os
import faiss
import requests
import numpy as np
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import faiss


LLM_MODEL = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class ChatBotBackend:
    def __init__(self, 
                 hf_api, 
                 model_id = LLM_MODEL, 
                 embedding_model= EMBEDDING_MODEL):
        self.hf_api = hf_api
        self.model_id = model_id
        self.embedding_model = embedding_model

        # API end point setup for model
        self.llm_api_url = f'https://api-inference.huggingface.co/models/{model_id}'

        # API endpoint for embedding model
        self.emb_api_url = f'https://api-inference.huggingface.co/pipeline/feature-extraction/{embedding_model}'

        # header
        self.headers = {
            "Authorization": f'Bearer {hf_api}', 
        }

    def get_embedding(self, chunks):
        response = requests.post(self.emb_api_url, headers=self.headers, json={'inputs': chunks})
        if response.status_code == 200:
            embeddings = response.json()
        else:
            raise Exception(f'Error fetching the data, {response.json()}')
        
        return embeddings
    
    def create_vector_store(self, chunks, metadatas):

        print(chunks)
        
        embeddings = self.get_embedding(chunks)
        dimension = len(embeddings[0])
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(embeddings, dtype=np.float32))
        
        document_index = (str(i): Document(page_content=chunks[0]))
        doc_store = InMemoryDocstore(document_index)
        index_to_docstore_id = {i: str(i) for i in range(len(chunks))}         



