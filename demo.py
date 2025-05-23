import torch
import numpy as np
import pandas as pd
import pickle
import difflib
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from natsort import natsorted
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from FlagEmbedding import FlagLLMReranker, LayerWiseFlagLLMReranker
from langchain_ollama import ChatOllama
import requests
import json

class PdfProcessor:
    def __init__(self):
        """Initialize PDF converter with pre-loaded model"""
        self.converter = PdfConverter(artifact_dict=create_model_dict())
    
    def process_document(self, doc_path, save_path = None):
        """Process document using pre-loaded model"""
        # Check if path is to a single file or a directory
        # Process a single file or all files in a directory
        all_texts = []
        
        # Use the file directly if it's a single file, otherwise get all PDFs in directory
        files = [doc_path] if doc_path.is_file() else list(doc_path.glob("*.pdf"))
        
        if save_path is not None:
            if not os.path.exists(save_path):
                # Create save path if it doesn't exist
                print(f"Creating save directory: {save_path}")
                os.makedirs(save_path)
        
        # Process each file
        for i, file in enumerate(files):
            # Skip if file already processed
            if os.path.exists(save_path / f"{file.stem}.md"):
                print(f"Skipping {file.stem} because .md already exists")
                # Load .md from disk
                with open(save_path / f"{file.stem}.md", "r") as f:
                    text = f.read()
                all_texts.append(text)
                continue
            
            print(f"Processing {file.stem} ({i+1}/{len(files)})")
            
            rendered_doc = self.converter(str(file))
            text, _, images = text_from_rendered(rendered_doc)
            all_texts.append(text)
            
            if save_path is not None: # save to .md's
                with open(save_path / f"{file.stem}.md", "w") as f:
                    f.write(text)
        
        # For single files, return the text as a list with one item
        return all_texts[0:1] if doc_path.is_file() else all_texts
        
class BiEncoderPipeline:
    _instances = {}  # Class variable to store instances
    
    def __new__(cls, 
                 model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
                 chunk_size=1024,
                 chunk_overlap=0):
        # Create a unique key for this model configuration
        instance_key = f"{model_name}_{chunk_size}_{chunk_overlap}"
        
        # If an instance with this configuration doesn't exist, create it
        if instance_key not in cls._instances:
            cls._instances[instance_key] = super(BiEncoderPipeline, cls).__new__(cls)
        
        return cls._instances[instance_key]
    
    def __init__(self, 
                 model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
                 chunk_size=1024,
                 chunk_overlap=0):
        # Skip initialization if this instance was already initialized
        if hasattr(self, 'initialized'):
            return
            
        print(f"Initializing BiEncoderPipeline with model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.initialized = True

    def embed_documents(self, doc_texts, doc_ids = None, save_path = None):
        """Embed documents using pre-loaded models"""
        # If string given (i.e., one document, big string), and not list (i.e., multiple documents or single document but list), make it a list
        if not isinstance(doc_texts, list):
            doc_texts = [doc_texts]
        
        if save_path is not None:
            actual_save_path = Path(save_path) / self.model_name.split("/")[-1] / f"chunk_size_{self.chunk_size}" / f"chunk_overlap_{self.chunk_overlap}"
            print(f"Actual save path: {actual_save_path}")
            # Create save path if it doesn't exist
            print(f"Creating save directory: {actual_save_path}")
            if not os.path.exists(actual_save_path):
                os.makedirs(actual_save_path)

        # Process each text in the list
        all_chunks = []
        all_vectors = []
        for i, doc in tqdm(enumerate(doc_texts), total=len(doc_texts), desc="Embedding documents"):
            # Create a directory for the document if it doesn't exist
            doc_dir = actual_save_path / doc_ids[i]
            if not os.path.exists(doc_dir):
                os.makedirs(doc_dir)
            
            # Split the document into chunks to check if all are already saved
            doc_chunks = self.text_splitter.split_text(doc)
            
            # Check if all chunks for this document are already saved
            if save_path is not None:
                all_chunks_exist = all(
                    os.path.exists(f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl") 
                    for j in range(len(doc_chunks))
                )
                
                if all_chunks_exist:
                    print(f"Skipping {doc_ids[i]} because all chunks already exist")
                    # Load existing chunks from disk
                    loaded_chunks = []
                    loaded_vectors = []
                    for j in range(len(doc_chunks)):
                        with open(f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl", "rb") as f:
                            chunk_data = pickle.load(f)
                            loaded_chunks.append(chunk_data["text"])
                            loaded_vectors.append(chunk_data["vector"])
                    
                    all_chunks.append(loaded_chunks)
                    all_vectors.append(loaded_vectors)
                    continue
            
            # Store the chunks and their embeddings
            all_chunks.append(doc_chunks)
            all_vectors.append(self.model.encode(doc_chunks))

        # Create results list; each element is a dict with the chunk text, its vector, its index, and the overall document index
        results = []
        for i, doc_chunks in enumerate(all_chunks): # For each document
            for j, chunk in enumerate(doc_chunks): # For each chunk
                results.append({
                    "text": chunk,
                    "vector": all_vectors[i][j],
                    "original_doc_id": doc_ids[i] if doc_ids is not None else None,
                    "doc_idx": i,
                    "chunk_idx": j
                })
            
                if save_path is not None:
                    # Only save if the chunk doesn't already exist
                    chunk_path = f"{actual_save_path}/{doc_ids[i]}/chunk_{j}.pkl"
                    if not os.path.exists(chunk_path):
                        with open(chunk_path, "wb") as f:
                            pickle.dump(results[-1], f)

        return results
    
    def retrieve_top_k(self, query, documents_embeddings, top_k=50):
        """Retrieve top k embeddings using cosine similarity to the query"""
        
        # Embed query
        query_vector = self.model.encode([query])   
        
        # Get embeddings from documents dicts
        stored_vectors = np.array([item["vector"] for item in documents_embeddings])
        
        # Compute similarities between query and stored vectors
        similarities = cosine_similarity(query_vector, stored_vectors).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                **documents_embeddings[i],
                "similarity": float(similarities[i])
                # "original_id": i
            }
            for i in top_indices
        ]
        
class CrossEncoderPipeline:
    _instances = {}  # Class variable to store instances
    
    def __new__(cls, model_name="cross-encoder/ms-marco-MiniLM-L6-v2", device=None):
        # Create a unique key for this model configuration
        instance_key = f"{model_name}_{device}"
        
        # If an instance with this configuration doesn't exist, create it
        if instance_key not in cls._instances:
            cls._instances[instance_key] = super(CrossEncoderPipeline, cls).__new__(cls)
        
        return cls._instances[instance_key]
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6-v2", device=None):
        # Skip initialization if this instance was already initialized
        if hasattr(self, 'initialized'):
            return
            
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.initialized = True
    
    def rerank(self, query, documents, top_n=4):
        """Rerank documents using cross-encoder"""
        
        # Get texts out of documents dicts
        texts = [doc["text"] for doc in documents]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            [query] * len(texts), # Repeat query for each document
            text_pair=texts, # Pair query with each document (i.e., query + chunk 1, query + chunk 2, ...)
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Compute logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Convert logits to scores
        scores = torch.sigmoid(logits).squeeze().cpu().numpy() # Convert to numpy array
        
        # Create results list
        results = []
        for idx, doc in enumerate(documents):
            results.append({
                **doc,
                "rerank_score": float(scores[idx])
            })
        
        results_sorted = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        return results_sorted[:top_n]


### DOCUMENT PROCESSING
# Process .pdf's into .md's
DOCUMENTS_DIR = Path("data/my_docs") # assumption: all documents are in one directory
PROCESSED_DOCS_DIR = Path("data/processed_docs")

# Check if there are any unprocessed documents
pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
unprocessed_files = [
    pdf_file for pdf_file in pdf_files 
    if not (PROCESSED_DOCS_DIR / f"{pdf_file.stem}.md").exists()
]

# Only initialize PdfProcessor if there are unprocessed documents
if unprocessed_files:
    print(f"Found {len(unprocessed_files)} unprocessed documents. Initializing PdfProcessor...")
    marker = PdfProcessor()
    processed_docs = marker.process_document(DOCUMENTS_DIR, save_path=PROCESSED_DOCS_DIR)
else:
    print("All documents are already processed. Loading existing .md files...")
    processed_docs = []
    for pdf_file in pdf_files:
        with open(PROCESSED_DOCS_DIR / f"{pdf_file.stem}.md", "r") as f:
            processed_docs.append(f.read())

docs_paths = [str(path.stem) for path in pdf_files] # assumption: all documents are .pdf's

## Embedding
# Embed .md's into vectors
bi_encoder = BiEncoderPipeline(chunk_size=2048, chunk_overlap=128) # uses snowflake-arctic-embed-l-v2.0 by default, but can be changed to any other embedding model
embeddings = bi_encoder.embed_documents(processed_docs, doc_ids = docs_paths, save_path = "data/embeddings/") # if given save_path, will save to working directory disk as .pkl


### RETRIEVAL
print("\nWelcome to the RAG Chat Demo! Type 'exit' to quit.")
print("----------------------------------------")

while True:
    # Get user query and clean it
    query = input("\nEnter your question (or 'exit' to quit): ").strip().replace('\r', '')
    
    # Check if user wants to exit
    if query.lower() == 'exit':
        print("\nGoodbye!")
        break
    
    if not query:
        print("Please enter a valid question.")
        continue
    
    print("\nRetrieving relevant documents...")
    ## Retrieval with BiEncoder
    # Retrieve top_k documents based on query
    retrieved_docs = bi_encoder.retrieve_top_k(query, embeddings, top_k=50)

    print("\nReranking documents...")
    ## Reranking with CrossEncoder
    # Rerank retrieved documents
    cross_encoder = CrossEncoderPipeline() # uses cross-encoder/ms-marco-MiniLM-L6-v2 by default, but can be changed to any other cross-encoder model
    reranked_docs = cross_encoder.rerank(query, retrieved_docs, top_n=8)

    print("\nGenerating response...")
    ### RESPONSE GENERATION
    llm = ChatOllama(model="cogito:3b")
    context_ids = [doc["original_doc_id"] for doc in reranked_docs]
    context_texts = [doc["text"] for doc in reranked_docs]
    context_texts_pretty = "\n".join([f"<DOCUMENT{i+1}: {context_ids[i]}>\nTEXT:\n{text}\n</DOCUMENT{i+1}: {context_ids[i]}>\n" for i, text in enumerate(context_texts)])

    rag_prompt = f"""
        <QUERY>
        {query}
        </QUERY>

        <INSTRUCTIONS>
        Answer the user's QUERY using the text in DOCUMENTS.
        Keep your answer grounded in the facts of the DOCUMENTS.
        Use the IDs of the DOCUMENTS in your response.
        If the DOCUMENTS don't contain the facts to answer the QUERY, your best response is "the materials do not appear to be sufficient to provide a good answer."
        </INSTRUCTIONS>
        
        <DOCUMENTS>
        {context_texts_pretty}
        </DOCUMENTS>
        """

    response = llm.invoke(rag_prompt)
    print("\nResponse:")
    print("----------------------------------------")
    print(response.content)
    print("----------------------------------------")