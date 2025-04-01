import torch
import numpy as np
import pandas as pd
import pickle
from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

class PdfProcessor:
    def __init__(self):
        """Initialize PDF converter with pre-loaded model"""
        
        self.converter = PdfConverter(artifact_dict=create_model_dict())
    
    def process_document(self, doc_path):
        """Process document using pre-loaded model"""
        # Check if path is to a single file or a directory
        if doc_path.is_file():
            # If it's a single file, process it
            rendered_doc = self.converter(str(doc_path))
            text, _, images = text_from_rendered(rendered_doc)
            text = [text]
            return text
        else:
            # If it's a directory, process all files in the directory
            all_texts = []
            for file in doc_path.glob("*.pdf"):
                rendered_doc = self.converter(str(file))
                text, _, images = text_from_rendered(rendered_doc)
                all_texts.append(text)
            return all_texts

class BiEncoderPipeline:
    def __init__(self, 
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size=500,
                 chunk_overlap=50):
        """Initialize BiEncoderPipeline with pre-loaded model"""
        
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    # def set_chunk_parameters(self, chunk_size=None, chunk_overlap=None):
    #     if chunk_size:
    #         self.chunk_size = chunk_size
    #     if chunk_overlap:
    #         self.chunk_overlap = chunk_overlap
    #     self.text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=self.chunk_size,
    #         chunk_overlap=self.chunk_overlap,
    #         length_function=len
    #     )

    def embed_documents(self, doc_text, doc_id = None):
        """Embed documents using pre-loaded models"""
        # If string given (i.e., one document, big string), and not list (i.e., multiple documents or single document but list), make it a list
        if not isinstance(doc_text, list):
            doc_text = [doc_text]

        # Process each text in the list
        all_chunks = []
        all_vectors = []
        for i, doc in enumerate(doc_text):
            # Split the document into chunks
            doc_chunks = self.text_splitter.split_text(doc)
            
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
                    "original_doc_id": doc_id[i] if doc_id is not None else None,
                    "doc_idx": i,
                    "chunk_idx": j
                })
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
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6-v2", device=None):
        """Initialize CrossEncoderPipeline with pre-loaded model"""
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
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

def quick_test(bi_encoder, cross_encoder, pdf_processor, doc_path, query, top_k=50, top_n=4):
    # Document processing
    document_text = pdf_processor.process_document(doc_path)

    # Embedding
    embedded_docs = bi_encoder.embed_documents(document_text)

    # Retrieval and reranking
    top_chunks = bi_encoder.retrieve_top_k(query, embedded_docs, top_k=top_k)
    reranked_results = cross_encoder.rerank(query, top_chunks, top_n=top_n)

    # Print reranked results, without the "vector" key
    results_df = pd.DataFrame(reranked_results)
    results_df = results_df.drop(columns=["vector"])
    print(results_df)
    

# Pipeline initialization
pdf_processor = PdfProcessor()
bi_encoder = BiEncoderPipeline()
cross_encoder = CrossEncoderPipeline()

# ###### Single document (.pdf) test #####
# # Document processing
# doc_path = Path("data/my_docs/ekbia2016.pdf")
# document_text = pdf_processor.process_document(doc_path)

# # Embedding
# embedded_docs = bi_encoder.embed_documents(document_text)

# # Retrieval and reranking
# query = "What is the point of a political economy perspective on HCI?"
# top_chunks = bi_encoder.retrieve_top_k(query, embedded_docs, top_k=25)
# reranked_results = cross_encoder.rerank(query, top_chunks)

# # Print reranked results, without the "vector" key
# results_df = pd.DataFrame(reranked_results)
# results_df = results_df.drop(columns=["vector"])
# print(results_df)
# ##### End of single document (.pdf) test #####


# ##### Multiple documents (.pdf) test #####
# # Document processing
# docs_path = Path("data/my_docs")
# docs_texts = pdf_processor.process_document(docs_path)

# # Embedding
# embedded_docs = bi_encoder.embed_documents(docs_texts)

# # Retrieval and reranking
# query = "What is the point of a political perspective on HCI?"
# top_chunks = bi_encoder.retrieve_top_k(query, embedded_docs, top_k=25)
# reranked_results = cross_encoder.rerank(query, top_chunks)

# # Print reranked results, without the "vector" key
# results_df = pd.DataFrame(reranked_results)
# results_df = results_df.drop(columns=["vector"])
# print(results_df)
# ##### End of multiple documents (.pdf) test #####

# # Quick test, single .pdf
# doc_path = Path("data/my_docs/ekbia2016.pdf")
# query = "What is the point of a political economy perspective on HCI?"
# quick_test(bi_encoder, cross_encoder, pdf_processor, doc_path, query)

# # Quick test, multiple .pdfs
# docs_path = Path("data/my_docs")
# query = "What is the point of a political perspective on HCI?"
# quick_test(bi_encoder, cross_encoder, pdf_processor, docs_path, query)


## RAGBench: TechQA, multiple documents test
## Load dataset
techqa_train = load_dataset("rungalileo/ragbench", "techqa", split="train").to_pandas()
techqa_val = load_dataset("rungalileo/ragbench", "techqa", split="validation").to_pandas()
techqa_test = load_dataset("rungalileo/ragbench", "techqa", split="test").to_pandas()

techqa = pd.concat([techqa_train, techqa_val, techqa_test], ignore_index=True)

# Filtering
techqa = techqa[techqa["generation_model_name"] == "gpt-3.5-turbo-0125"] # The authors tested two models, we only want the results for gpt-3.5-turbo-0125
techqa = techqa[["id", "question", "documents", "documents_sentences", "dataset_name", "all_relevant_sentence_keys", "all_utilized_sentence_keys"]]
techqa = techqa.rename(columns={"id": "question_id"})

# Redo id's
techqa = techqa.sample(frac=1, random_state=1).reset_index(drop=True)
techqa["question_id"] = techqa.index


##### Documents DataFrame
# Create a new dataframe with each document as a separate row
techqa_exp = techqa.explode('documents').reset_index(drop=True)

# Create a new 'doc_id' column that combines question_id with document number
techqa_exp['doc_id'] = techqa_exp.groupby('question_id').cumcount() + 1
techqa_exp['doc_id'] = techqa_exp['question_id'].apply(lambda x: f'{x}') + '-' + techqa_exp['doc_id'].apply(lambda x: f'doc{x}')

# Keep only the 'documents' and 'doc_id' columns
techqa_exp = techqa_exp[["documents", "doc_id"]]

## Find duplicates
# Sort documents alphabetically
techqa_exp.sort_values(by='documents', inplace=True)
techqa_exp.reset_index(drop=True, inplace=True)

# Add a 'duplicated' column
techqa_exp["duplicated"] = False

# Compare each document with the next one
for i in range(len(techqa_exp)-1):
    if techqa_exp.loc[i, "documents"] == techqa_exp.loc[i+1, "documents"]:
        techqa_exp.loc[i, "duplicated"] = True
        techqa_exp.loc[i+1, "doc_id"] = "_".join([techqa_exp.loc[i, "doc_id"],
                                                       techqa_exp.loc[i+1, "doc_id"]])
        
# Split doc_id column by "_"
techqa_exp["doc_id"] = techqa_exp["doc_id"].str.split("_")
        
# Drop duplicates
techqa_exp = techqa_exp[techqa_exp["duplicated"] == False]
techqa_exp = techqa_exp.drop(columns=["duplicated"])



#### TECHQA EMBEDDING
bi_encoder_1000_100 = BiEncoderPipeline(chunk_size=1000,
                                        chunk_overlap=100)

# Embed documents
techqa_embed = bi_encoder_1000_100.embed_documents(techqa_exp.documents.to_list(),
                                                   techqa_exp.doc_id.to_list())

# Save embeddings
output_path = Path("techqa_embeddings/MiniLM-L6-v2")
output_path.mkdir(parents=True, exist_ok=True)
with open(output_path / "size-1000_overlap-100.pkl", "wb") as f:
    pickle.dump(techqa_embed, f)

# ## MATCH CHUNKS AND TECHQA SENTENCES
# # Get text out of every dict in the list
# for item in techqa_embed:
#     chunk = item["text"]
#     search_id = item["original_doc_id"]
    
#     # FIND ROWS IN TECHQA THAT CONTAIN ANY OF THE ENTRIES IN SEARCH_ID

#     # Get rows from techqa dataframe that contain one of the doc_ids
#     matching_rows = techqa_exp[techqa_exp['doc_id'].isin(["129-doc1"])]
    


