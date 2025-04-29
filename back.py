import os
import faiss
import requests
import numpy as np
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


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

    def get_embedding(self, chunks, batch_size=10):
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            response = requests.post(self.emb_api_url, headers=self.headers, json={'inputs': batch})
            if response.status_code == 200:
                embeddings.extend(response.json())
            else:
                raise Exception(f'Error fetching the data, {response.json()}')
        return embeddings
    
    def create_vector_store(self, chunks, metadatas):

        print(chunks)
        
        embeddings = self.get_embedding(chunks)
        dimension = len(embeddings[0])
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(embeddings, dtype=np.float32))
        
        documents_dict = {str(i): Document(page_content=chunks[i], metadata=metadatas[i]) for i in range(len(chunks))}
        doc_store = InMemoryDocstore(documents_dict)
        index_to_docstore_id = {i: str(i) for i in range(len(chunks))}        

        self.vector_store = FAISS(
            index=faiss_index,
            docstore=doc_store,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function = lambda x: self.get_embedding(x)
        ) 

    def retrieve_context(self, query, chunks):
        query_embedding = self.get_embedding([query])[0]
        distance, indices = self.vector_store.index.search(np.array([query_embedding], dtype=np.float32), k=3)
        retreived_texts = [chunks[i] for i in indices[0] if i < len(chunks)]

        return "\n\n".join(retreived_texts)

    def answer_prompt(self, prompt):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 512, 
                "min_length": 30, 
                "temperature": 0.5, 
                "num_return_sequences": 1
            }
        }  

        response = requests.post(self.llm_api_url, headers=self.headers, json=payload)

        if response.status_code == 200:
            try:
                response_json = response.json()
                return (response_json[0])
            
            except Exception as e:
                print(f'Error: {e}')
                print(f'raw response: {response.text}')
                return f'Error: {str(e)}'
        else: 
            error_message = f'Error ({response.status_code}): {response.text}'
            return error_message
        
    def get_answer(self, query, chunks, use_context=True):
        context = self.retrieve_context(query, chunks)

        prompt = f"""Based on the following information, please answer question throughly.
        Information:
        {context}

        Question:
        {query}

        """
        response = self.answer_prompt(prompt)
        
        return response['generated_text']