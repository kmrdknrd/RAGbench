import torch
import numpy as np
import pandas as pd
import pickle
import difflib
import re
import seaborn as sns
import matplotlib.pyplot as plt
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
import requests
import json

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
                 model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
                 chunk_size=1024,
                 chunk_overlap=0):
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

class FlagEmbeddingReranker:
    def __init__(
        self, 
        model_name="BAAI/bge-reranker-v2-gemma", 
        use_fp16=True, 
        use_bf16=False
    ):
        """
        Initialize with FlagLLMReranker model
        
        Args:
            model_name: Name of the reranker model
            use_fp16: Use FP16 precision (faster with slight performance loss)
            use_bf16: Use BF16 precision (alternative to FP16)
        """
        if use_bf16:
            self.reranker = LayerWiseFlagLLMReranker(model_name, use_bf16=True)
        else:
            self.reranker = LayerWiseFlagLLMReranker(model_name, use_fp16=use_fp16)
    
    def rerank(self, query, documents, top_n=4):
        """Rerank documents using FlagLLMReranker"""
        # Extract texts and create query-passage pairs
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Compute scores for all pairs
        scores = self.reranker.compute_score(pairs, cutoff_layers=[28])[0]
        
        # Add scores to documents and sort
        results = [
            {**doc, "rerank_score": float(score)} 
            for doc, score in zip(documents, scores)
        ]
        
        # Return top results
        return sorted(
            results, 
            key=lambda x: x['rerank_score'], 
            reverse=True
        )[:top_n]
    
class LocalBiEncoderPipeline:
    def __init__(self, 
                 model_name="text-embedding-bge-m3",
                 chunk_size=500,
                 chunk_overlap=50,
                 local_server_url="http://127.0.0.1:1234"):
        """Initialize BiEncoderPipeline with local model"""
        
        self.model_name = model_name
        self.local_server_url = local_server_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def embed_documents(self, doc_text, doc_id=None):
        """Embed documents using local model"""
        # If string given (i.e., one document, big string), and not list (i.e., multiple documents or single document but list), make it a list
        if not isinstance(doc_text, list):
            doc_text = [doc_text]

        # Process each text in the list
        all_chunks = []
        all_vectors = []
        for i, doc in enumerate(doc_text):
            # Split the document into chunks
            doc_chunks = self.text_splitter.split_text(doc)
            
            # Store the chunks
            all_chunks.append(doc_chunks)
            
            # Get embeddings from local server
            embeddings = []
            for chunk in doc_chunks:
                response = requests.post(
                    f"{self.local_server_url}/v1/embeddings",
                    json={
                        "model": self.model_name,
                        "input": chunk
                    }
                )
                if response.status_code == 200:
                    embedding = response.json()["data"][0]["embedding"]
                    embeddings.append(embedding)
                else:
                    raise Exception(f"Error getting embeddings: {response.text}")
            
            all_vectors.append(np.array(embeddings))

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
        
        # Embed query using local model
        response = requests.post(
            f"{self.local_server_url}/v1/embeddings",
            json={
                "model": self.model_name,
                "input": query
            }
        )
        if response.status_code == 200:
            query_vector = np.array(response.json()["data"][0]["embedding"]).reshape(1, -1)
        else:
            raise Exception(f"Error getting embeddings: {response.text}")
        
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
    
    
def retrieve_and_rerank(queries, embeddings, bi_encoder, cross_encoder, dataset, top_k=50, save_results=False, save_path=None):
    """
    Retrieves and reranks chunks for a list of queries.
    
    Args:
        queries: List of query strings or DataFrame with 'question' column
        embeddings: Embedded chunks to search through
        bi_encoder: Bi-encoder model for initial retrieval
        cross_encoder: Cross-encoder model for reranking
        dataset: DataFrame containing metadata about queries
        top_k: Number of top chunks to retrieve (default: 50)
        save_results: Whether to save the results list as a pickle file (default: False)
        save_path: Path to save the results list (default: None, which saves to current directory)
        
    Returns:
        list: List of dictionaries containing retrieval results
    """
    # Handle different input types
    if not isinstance(queries, list):
        queries = queries.question.tolist()
    
    results_list = []
    for i, query in tqdm(enumerate(queries), total=len(queries), desc="Processing queries"):
        top_chunks = bi_encoder.retrieve_top_k(query, embeddings, top_k=top_k)  # Retrieve top chunks for each query
        reranked_results = cross_encoder.rerank(query, top_chunks)  # Rerank the chunks
        reranked_results = [{k: v for k, v in d.items() if k not in ["vector", "match_types"]} for d in reranked_results]
        
        expected_sentences = dataset.loc[i, "all_relevant_sentence_keys"]
        
        full_results = {"query": query,
                        "question_id": dataset.loc[i, "question_id"],
                        "expected": expected_sentences,
                        "results": reranked_results}
        results_list.append(full_results)
    
    # Save results if requested
    if save_results:
        # Error if save_path is not provided
        if save_path is None:
            raise ValueError("save_path must be provided if save_results is True")
        
        # Create directory if it doesn't exist
        if not Path(save_path).parent.exists() and str(Path(save_path).parent) != ".":
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
        with open(save_path, "wb") as f:
            pickle.dump(results_list, f)
        print(f"Results saved to {save_path}")
    
    return results_list
  
        
def compute_recall(results_list):
    """
    Compute recall for a list of retrieval results.
    
    Args:
        results_list: List of dictionaries containing retrieval results
        
    Returns:
        float: Mean recall score (excluding NaN values)
        np.array: Array of recall scores for each query
    """
    recalls = np.array([])
    for result in results_list:
        # Get question id and expected sentences
        q_id = result["question_id"]
        expected_sentences = result["expected"]
        
        # If no expected sentences, skip
        if len(expected_sentences) == 0:
            recall = np.nan
            recalls = np.append(recalls, recall)
            continue
        
        # Get actual results
        results_sentences = [match for matches in result["results"] for match in [matches["sentence_matches"]]]
        results_doc_ids = [id for ids in result["results"] for id in [ids["original_doc_id"]]]
        
        # Get rid of numbers in results_sentences
        results_sentences = [[re.sub(r'\d+', '', s) for s in sublist] for sublist in results_sentences]
        
        # Get expected doc_ids
        expected_doc_ids = np.unique([int(re.search(r'\d+', item).group()) for item in expected_sentences]) + 1 # Convert to 1-indexed. Expected sentences are 0-indexed, but doc_ids are 1-indexed
        expected_doc_ids = [f"{q_id}-doc{doc}" for doc in expected_doc_ids] # Convert to doc-id format

        # Check if expected doc_ids are in results_doc_ids, and if so, get the sentences that match
        already_matched = {}
        positives = 0
        for i, e_doc in enumerate(expected_doc_ids):
            for j, r_doc in enumerate(results_doc_ids):
                if e_doc in r_doc:
                    
                    results_sentences_for_matching = results_sentences[j].copy()
                    
                    if e_doc in already_matched:
                        # If a sentence has already been matched, omit it (possible because of chunking overlap)
                        results_sentences_for_matching = [s for s in results_sentences_for_matching if s not in already_matched[e_doc]]
                        
                    # Omit irrelevant expected sentences from comparison (so if e_doc = 1-doc1, omit expected_sentences that didn't come from 1-doc1)
                    # i.e., sentences that don't start with last digit of e_doc
                    expected_sentences_for_matching = [s for s in expected_sentences if s.startswith(str(int(e_doc[-1]) - 1))] # Sentences are 0-indexed, doc_ids are 1-indexed
                    expected_sentences_for_matching = [s[1:] for s in expected_sentences_for_matching]
                    
                    # get intersection of expected and results_sentences_for_matching
                    intersection = set(expected_sentences_for_matching) & set(results_sentences_for_matching)
                    positives += len(intersection)
                    
                    # Add matched sentences to already_matched
                    if e_doc in already_matched:
                        already_matched[e_doc].extend(results_sentences_for_matching)
                    else:
                        already_matched[e_doc] = results_sentences_for_matching
        
        # Compute recall
        recall = positives / len(expected_sentences)
        recalls = np.append(recalls, recall)
        
    return np.nanmean(recalls), recalls


# Pipeline initialization
# pdf_processor = PdfProcessor()
# bi_encoder = BiEncoderPipeline()
cross_encoder = CrossEncoderPipeline(model_name="Alibaba-NLP/gte-reranker-modernbert-base")


##################### TECHQA #####################
## Load dataset
# techqa_train = load_dataset("rungalileo/ragbench", "techqa", split="train").to_pandas()
# techqa_val = load_dataset("rungalileo/ragbench", "techqa", split="validation").to_pandas()
# techqa_test = load_dataset("rungalileo/ragbench", "techqa", split="test").to_pandas()

# techqa = pd.concat([techqa_train, techqa_val, techqa_test], ignore_index=True)

# # Filtering
# techqa = techqa[techqa["generation_model_name"] == "gpt-3.5-turbo-0125"] # The authors tested two models, we only want the results for gpt-3.5-turbo-0125
# techqa = techqa[["id", "question", "documents", "documents_sentences", "dataset_name", "all_relevant_sentence_keys", "all_utilized_sentence_keys"]]
# techqa = techqa.rename(columns={"id": "question_id"})

# # Redo id's
# techqa = techqa.sample(frac=1, random_state=1).reset_index(drop=True)
# techqa["question_id"] = techqa.index

# # Save techqa
# with open("techqa.pkl", "wb") as f:
#     pickle.dump(techqa, f)

# Load techqa
with open("techqa.pkl", "rb") as f:
    techqa = pickle.load(f)

# ##### GET DOCUMENTS, REMOVE DUPLICATES #####
# # Create a new dataframe with each document as a separate row
# techqa_exp = techqa.explode(list(('documents', 'documents_sentences'))).reset_index(drop=True)

# # Create a new 'doc_id' column that combines question_id with document number
# techqa_exp['doc_id'] = techqa_exp.groupby('question_id').cumcount() + 1
# techqa_exp['doc_id'] = techqa_exp['question_id'].apply(lambda x: f'{x}') + '-' + techqa_exp['doc_id'].apply(lambda x: f'doc{x}')

# # Keep only the 'documents', 'doc_id', and 'documents_sentences' columns
# techqa_exp = techqa_exp[["documents", "doc_id", "documents_sentences"]]

# ## Find duplicates
# # Sort documents alphabetically
# techqa_exp.sort_values(by='documents', inplace=True)
# techqa_exp.reset_index(drop=True, inplace=True)

# # Add a 'duplicated' column
# techqa_exp["duplicated"] = False

# # Compare each document with the next one
# for i in range(len(techqa_exp)-1):
#     if techqa_exp.loc[i, "documents"] == techqa_exp.loc[i+1, "documents"]:
#         techqa_exp.loc[i, "duplicated"] = True
#         techqa_exp.loc[i+1, "doc_id"] = "_".join([techqa_exp.loc[i, "doc_id"],
#                                                        techqa_exp.loc[i+1, "doc_id"]])
        
# # Split doc_id column by "_"
# techqa_exp["doc_id"] = techqa_exp["doc_id"].str.split("_")
        
# # Drop duplicates
# techqa_exp = techqa_exp[techqa_exp["duplicated"] == False]
# techqa_exp = techqa_exp.drop(columns=["duplicated"])
# techqa_exp = techqa_exp.reset_index(drop=True)

# # Save techqa_exp
# with open("techqa_exp.pkl", "wb") as f:
#     pickle.dump(techqa_exp, f)

# Load techqa_exp
with open("techqa_exp.pkl", "rb") as f:
    techqa_exp = pickle.load(f)

# ##### TECHQA EMBEDDING #####
# bi_encoder_1000_0 = BiEncoderPipeline(chunk_size=1000,
#                                       chunk_overlap=0)

# # Embed documents
# techqa_embed = bi_encoder_1000_0.embed_documents(techqa_exp.documents.to_list(),
#                                                  techqa_exp.doc_id.to_list())

# # Save embeddings
# output_path = Path("techqa_embeddings/MiniLM-L6-v2")
# output_path.mkdir(parents=True, exist_ok=True)
# with open(output_path / "size-1000_overlap-0.pkl", "wb") as f:
#     pickle.dump(techqa_embed, f)
    
# # Load embeddings
# with open(output_path / "size-1000_overlap-100.pkl", "rb") as f:
#     techqa_embed = pickle.load(f)

techqa_questions = techqa.question.tolist()
chunk_size = [4096, 8192]
chunk_overlap = [0, 128]
for c_size in chunk_size:
    for c_overlap in chunk_overlap:
        print(f"c_size: {c_size}, c_overlap: {c_overlap}")
        
        # if c_size == 1024 and c_overlap == 0:
        #     continue
        
        bi_encoder_text_embedding = BiEncoderPipeline(
            model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
            chunk_size=c_size,
            chunk_overlap=c_overlap
            )
        
        techqa_embed = bi_encoder_text_embedding.embed_documents(techqa_exp.documents.to_list(),
                                                                 techqa_exp.doc_id.to_list())
        
        wiggle_room = 1 if c_overlap > 0 else 0
        previous_doc_idx = None
        size_zero_counter = 0
        m_counter = 0
        for i, item in enumerate(techqa_embed):
            chunk = item["text"] # The chunk to match
            search_id = item["original_doc_id"][0] # The document id where the chunk comes from
            doc_idx = item["doc_idx"]
            techqa_embed[i]["sentence_matches"] = []
            techqa_embed[i]["match_types"] = []
            
            print(f"\n\nchunk: {item['chunk_idx']}")
            
            # Remove punctuation from chunk
            chunk_no_punct = ''.join(e for e in chunk if e.isalnum())
            
            # Find document in techqa_exp that the chunk comes from, get the sentences
            doc_sentences = techqa_exp[techqa_exp['doc_id'].apply(lambda x: search_id in x)].documents_sentences.tolist()[0]
            
            # Create a dictionary of the sentences (key: sentence key, value: sentence text), and list of keys
            doc_dict = {arr[0]: arr[1] for arr in doc_sentences}
            doc_dict_keys = list(doc_dict.keys())
            
            # If doc_idx is different from the previous doc_idx, reset the last_match_key
            if doc_idx != previous_doc_idx:
                last_match_key = None
                previous_doc_idx = doc_idx
            
            # For each sentence in the document, check if the chunk contains it
            no_match_counter = 0 # Counter for # of times a sentence does not match the conditions
            sentence_idx = -1
            for key, sentence in doc_dict.items():
                # Skip all sentences until the one before the last match (if last key is 3d, skip all sentences until key is 3c)
                # You might think if last match was 3d in the previous chunk, then we should skip until 3d for the current chunk, but because of chunking overlap, the current chunk might still have 3c in it
                if last_match_key is not None:
                    if doc_dict_keys.index(key) < last_match_key - wiggle_room:
                        print(f"skipping key: {key}")
                        continue
                
                sentence_idx += 1
                        
                print(f"\nchecking key: {key}")
                sentence_no_punct = ''.join(e for e in sentence if e.isalnum())
                
                if len(chunk_no_punct) == 0 or len(sentence_no_punct) == 0:
                    no_match_counter = 0 # Reset the no_match_counter if the chunk or sentence is empty for wiggle room
                
                # Match sentence with chunk
                s = difflib.SequenceMatcher(None,
                                            chunk_no_punct,
                                            sentence_no_punct,
                                            autojunk=False)
                
                # Find the longest match
                pos_a, pos_b, size = s.find_longest_match(0, len(chunk_no_punct),
                                                          0, len(sentence_no_punct))
                # pos_a is the start index of the match in the chunk
                # pos_b is the start index of the match in the sentence
                # size is the length of the match
                # len(pos_a:pos_a+size) = len(pos_b:pos_b+size)
                
                # For the first two sentences, skip if the match is not at the beginning of the chunk 
                if sentence_idx < 2 and pos_a > size:
                    m_counter += 1
                    continue
                
                matching_part = chunk_no_punct[pos_a:pos_a+size]
                
                # Skip if no match
                if size == 0:
                    size_zero_counter += 1
                    no_match_counter += 1
                    if no_match_counter > 2:
                        break
                    continue
            
                
                ## Check conditions        
                sentence_100_match = size == len(sentence_no_punct)
                if sentence_100_match:
                    techqa_embed[i]["sentence_matches"].append(key)
                    last_match_key = doc_dict_keys.index(key)
                    
                    # Remove first instance of matching part
                    chunk_no_punct = chunk_no_punct.replace(matching_part, "", 1)
                    continue
                
                chunk_100_match = size == len(chunk_no_punct)
                if chunk_100_match:
                    techqa_embed[i]["sentence_matches"].append(key)
                    last_match_key = doc_dict_keys.index(key)
                    break # If the sentence contains the whole chunk, then the next sentences will not contain any more matches
        
                contains_start = pos_a == 0 # The match starts at the beginning of the chunk (i.e., no text in the chunk before the match)
                no_text_after = pos_b + size == len(sentence_no_punct) # The match is at the end of the sentence (i.e., no text in the sentence after the match)
                matching_ratio = len(sentence_no_punct[pos_b:pos_b+size]) / len(sentence_no_punct) # Portion of the sentence that contains the match
                if contains_start and no_text_after and matching_ratio >= 0.5:
                    techqa_embed[i]["sentence_matches"].append(key)
                    last_match_key = doc_dict_keys.index(key)
                    
                    chunk_no_punct = chunk_no_punct.replace(matching_part, "", 1)
                    continue
                    
                contains_end = pos_a + size == len(chunk_no_punct) # The match is at the end of the chunk (i.e., no text in the chunk after the match)
                no_text_before = pos_b == 0 # The match starts at the beginning of the sentence (i.e., no text in the sentence before the match)
                if contains_end and no_text_before and matching_ratio >= 0.5:
                    techqa_embed[i]["sentence_matches"].append(key)
                    last_match_key = doc_dict_keys.index(key)
                    
                    chunk_no_punct = chunk_no_punct.replace(matching_part, "", 1)
                    continue
                
                else:
                    if sentence_idx > 0: # Because we go back an extra key for the previous chunk, don't count no_match_counter for the first sentence
                        no_match_counter += 1
                        if no_match_counter > 2:
                            break
                    continue
                    
                    
        techqa_embed_final = techqa_embed
        with open(f"techqa_embeddings/Snowflake/size-{c_size}_overlap-{c_overlap}_final.pkl", "wb") as f:
            pickle.dump(techqa_embed_final, f)
            
            
        # Load embeddings
        with open(f"techqa_embeddings/Snowflake/size-{c_size}_overlap-{c_overlap}_final.pkl", "rb") as f:
            techqa_embed_final = pickle.load(f)
            
        results_list = []
        for i, query in tqdm(enumerate(techqa.question), total=len(techqa.question), desc="Processing queries"):
            top_chunks = bi_encoder_text_embedding.retrieve_top_k(query, techqa_embed_final, top_k=50) # Retrieve top 50 chunks for eachq query
            reranked_results = cross_encoder.rerank(query, top_chunks) # Rerank the chunks
            reranked_results = [{k: v for k, v in d.items() if k not in ["vector", "match_types"]} for d in reranked_results]
            
            expected_sentences = techqa.loc[i, "all_relevant_sentence_keys"]
            
            full_results = {"query": query,
                            "question_id": techqa.loc[i, "question_id"],
                            "expected": expected_sentences,
                            "results": reranked_results}
            results_list.append(full_results)
            
        # Save
        with open(f"techqa_results/Snowflake/gte-cross-encoder_results_list_size-{c_size}_overlap-{c_overlap}.pkl", "wb") as f:
            pickle.dump(results_list, f)
        
        
## Match chunks and TechQA sentences
# READ TO UNDERSTAND (CONDITIONS):
# 1. Given a text chunk and a sentence we want to match (two strings), the sentence must either fully be or contain a substring of the chunk.
# 2. If the sentence is entirely a substring of the chunk, then the sentence is a 100% match.
# 3a. If the sentence is not entirely a substring of the chunk, but contains part of it, then the sentence is a partial match.
# 3b. A partial match must contain either part of the start or the end of the chunk. If the sentence contains part of the start of a chunk (start match; but not the whole chunk), there can be no text after the matching part. Likewise, if it contains part of the end of a chunk (end match; but not the whole chunk), there can be not text before the matching part.
# 3c. A partial match cannot contain irrelevant text before AND after the matching part. Before OR after is alright
# Explanation of 3c: Say the chunk is "def ghi jkl mno", and the sentence is "abc def ghi 123". The sentence is a partial match, contains the start of the chunk ("def ghi"), but it contains other text before and after it ("abc" and "123").
# Explanation cont.: Since it contains other text before AND after the matching part, the sentence could NOT have come from the same part of the document as the chunk did.
# 4. If the sentence is bigger than the chunk and the chunk is fully contained in the sentence, then the math works out (just the other way around), and the sentence is still labeled as a match.


########## BENCHMARK TECHQA ##########
# Compute recall mean, ignore nan's
results_list = pickle.load(open("techqa_results/Snowflake/gte-cross-encoder_results_list_size-1024_overlap-0.pkl", "rb"))
recall_mean, recalls = compute_recall(results_list)
print(f"Recall mean: {recall_mean}")

# Plot BGE, JINA, MXBAI, NOMIC and Snowflake results together (1024, 2048, 4096)
# Load results
# Plot results


overlap_0_results = np.array([])
overlap_100_results = np.array([])
for results_file in natsorted(Path("techqa_results/BGE-M3").glob("results_list_bge_size-*_overlap-*.pkl")):
    # skip if 8000
    if "8000" in str(results_file):
        continue
    
    results_list = pickle.load(open(results_file, "rb"))
    recall_mean, recalls = compute_recall(results_list)
    if "overlap-0" in str(results_file):
        overlap_0_results = np.append(overlap_0_results, recall_mean)
    else:
        overlap_100_results = np.append(overlap_100_results, recall_mean)
        
results_df = pd.DataFrame({"chunk_size": ["1024", "2048", "4096"],
                           "overlap_0": overlap_0_results,
                           "overlap_100": overlap_100_results})

results_df = results_df.melt(id_vars=["chunk_size"], var_name="overlap", value_name="recall")
results_df["recall"] = results_df["recall"] * 100

# Bar plot where recall is on the y-axis, and chunk size is on the x-axis, with two bars for each chunk size: one for overlap 0 and one for overlap 100
ax = sns.barplot(x="chunk_size", y="recall", hue = "overlap", data=results_df)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles, title="Chunk overlap", labels=["0 tokens", "100 tokens"], loc="upper left")
ax.set_xlabel("Chunk size")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel("Recall (%)")
ax.set_ylim(0, 70)

# add title
ax.set_title("BGE-M3")
# Show max recall value in the plot
max_recall = results_df["recall"].max()
ax.text(results_df.loc[results_df["recall"] == max_recall, "chunk_size"].values[0], max_recall + 5, f"{max_recall:.2f}%", ha="center", va="bottom")

# # Add vertical line between 1000 and 1250 (AnythingLLM chunk size limit)
# x_ticks = ax.get_xticks()
# x_labels = [label.get_text() for label in ax.get_xticklabels()]
# idx_1000 = x_labels.index("1000")
# idx_1250 = x_labels.index("1250")
# line_x = (x_ticks[idx_1000] + x_ticks[idx_1250]) / 2
# ax.axvline(x=line_x, color='red', linestyle='--')
# ax.text(line_x + 0.1, 45, "AnythingLLM chunk size limit", rotation=90, color='red')

# Save plot as pdf
plt.savefig(f"recall_barplot_BGE-M3.pdf", dpi=300)


print(f"Overlap 0 recalls:   {', '.join([f'{x:.2f}%' for x in np.round(overlap_0_results, 4)*100])}")
print(f"Overlap 100 recalls: {', '.join([f'{x:.2f}%' for x in np.round(overlap_100_results, 4)*100])}")








# # # test 1: big chunk, small sentence, small irrelevant start, should pass
# # chunk = 'IBM Support Portal [http://www.ibm.com/support] Machine Code License and Licensed Internal Code [http://www.ibm.com/support/docview.wss?uid=isg3T1025362] Machine warranties and license information [http://www.ibm.com/support/docview.wss?uid=isg3T1025361] International Warranty Service [http://www.ibm.com/support/docview.wss?uid=isg3T1025366] Advanced Part Exchange Warranty Service [http://www.ibm.com/support/docview.wss?uid=isg3T1025367] Authorized Use Table [http://www.ibm.com/support/docview.wss?uid=isg3T1025368] Environmental notices [http://www.ibm.com/support/docview.wss?uid=isg3T1025370] Install Policy [http://www.ibm.com/support/docview.wss?uid=isg3T1025365] Terms by product [http://www.ibm.com/support/docview.wss?uid=isg3T1025369] FAQs [http://www.ibm.com/support/docview.wss?uid=isg3T1025364] Glossary [http://www.ibm.com/support/docview.wss?uid=isg3T1025363] [data:image/gif;base64,R0lGODlhEAABAPAAAAAAAP///yH5BAEAAAEALAAAAAAQAAEAQAIEjI8ZBQA7]'
# # test = "irrelevant text IBM Support Portal [http://www.ibm.com/support] Machine Code License"
    
# # s = difflib.SequenceMatcher(None,
# #                             chunk,
# #                             test)

# # pos_a, pos_b, size = s.find_longest_match(0, len(chunk), 0, len(test))
# # print(pos_a, pos_b, size)
# # print(pos_a+size, pos_b+size)
# # print(len(chunk), len(test))

# # print(chunk[pos_a:pos_a+size])
# # print(test[pos_b:pos_b+size])

# # s.ratio()

# # # test 2: big chunk, small sentence, irrelevant start and end (middle matches only), should fail
# # chunk = chunk
# # test = "irrelevant text Support Portal [http://www.ibm.com/support] Machine Code License irrelevant text"

# # s = difflib.SequenceMatcher(None,
# #                             chunk,
# #                             test)

# # pos_a, pos_b, size = s.find_longest_match(0, len(chunk), 0, len(test))
# # print(pos_a, pos_b, size)
# # print(pos_a+size, pos_b+size)
# # print(len(chunk), len(test))

# # print(chunk[pos_a:pos_a+size])
# # print(test[pos_b:pos_b+size])

# # matching_ratio = len(chunk[pos_a:pos_a+size]) / len(test)
# # print(matching_ratio)

# # # test 3: big chunk, small sentence, contained within, should pass
# # chunk = chunk
# # test = "[http://www.ibm.com/support] Machine Code License and Licensed Internal"

# # s = difflib.SequenceMatcher(None,
# #                             chunk,
# #                             test)

# # pos_a, pos_b, size = s.find_longest_match(0, len(chunk), 0, len(test))
# # print(pos_a, pos_b, size)
# # print(pos_a+size, pos_b+size)
# # print(len(chunk), len(test))

# # print(chunk[pos_a:pos_a+size])
# # print(test[pos_b:pos_b+size])

# # matching_ratio = len(chunk[pos_a:pos_a+size]) / len(test)
# # print(matching_ratio)

# # # test 4: big chunk, small sentence, small irrelevant end, should pass
# # chunk = chunk
# # test = "[data:image/gif;base64,R0lGODlhEAABAPAAAAAAAP///yH5BAEAAAEALAAAAAAQAAEAQAIEjI8ZBQA7] small irrelevant end"

# # s = difflib.SequenceMatcher(None,
# #                             chunk,
# #                             test)

# # pos_a, pos_b, size = s.find_longest_match(0, len(chunk), 0, len(test))
# # print(pos_a, pos_b, size)
# # print(pos_a+size, pos_b+size)
# # print(len(chunk), len(test))

# # print(chunk[pos_a:pos_a+size])
# # print(test[pos_b:pos_b+size])

# # matching_ratio = len(chunk[pos_a:pos_a+size]) / len(test)
# # print(matching_ratio)

# # # test 5: big chunk, bigger sentence, small irrelevant start andend, should pass
# # chunk = chunk
# # test = "irrelevant text " + chunk + " more irrelevant text"

# # s = difflib.SequenceMatcher(None,
# #                             chunk,
# #                             test)

# # pos_a, pos_b, size = s.find_longest_match(0, len(chunk), 0, len(test))
# # print(pos_a, pos_b, size)
# # print(pos_a+size, pos_b+size)
# # print(len(chunk), len(test))

# # print(chunk[pos_a:pos_a+size])
# # print(test[pos_b:pos_b+size])

# # matching_ratio = len(chunk[pos_a:pos_a+size]) / len(test)
# # print(matching_ratio)

# # # test 6: big chunk, small sentence, big irrelevant end, should fail
# # chunk = chunk
# # test = "[data:image/gif;base64,R0lGODlhEAABAPAAAAAAAP///yH5BAEAAAEALAAAAAAQAAEAQAIEjI8ZBQA7] big irrelevant end big irrelevant end big irrelevant end big irrelevant end big irrelevant end big irrelevant end big irrelevant end"

# # s = difflib.SequenceMatcher(None,
# #                             chunk,
# #                             test)

# # pos_a, pos_b, size = s.find_longest_match(0, len(chunk), 0, len(test))
# # print(pos_a, pos_b, size)
# # print(pos_a+size, pos_b+size)
# # print(len(chunk), len(test))

# # print(chunk[pos_a:pos_a+size])
# # print(test[pos_b:pos_b+size])

# # matching_ratio = len(chunk[pos_a:pos_a+size]) / len(test)
# # print(matching_ratio)

# # # test 7: big chunk, small sentence, totally irrelevant, should fail
# # chunk = chunk
# # test = "totally irrelevant text IBM not"

# # s = difflib.SequenceMatcher(None,
# #                             chunk,
# #                             test)

# # pos_a, pos_b, size = s.find_longest_match(0, len(chunk), 0, len(test))
# # print(pos_a, pos_b, size)
# # print(pos_a+size, pos_b+size)
# # print(len(chunk), len(test))

# # print(chunk[pos_a:pos_a+size])
# # print(test[pos_b:pos_b+size])

# # matching_ratio = len(chunk[pos_a:pos_a+size]) / len(test)
# # print(matching_ratio)

# # # test 8: small chunk, big sentence, big irrelevant start and end, should pass
# # chunk = "ssss small chunk"
# # test = "really really irrelevant text small chunk more really really irrelevant text"

# # s = difflib.SequenceMatcher(None,
# #                             chunk,
# #                             test)

# # pos_a, pos_b, size = s.find_longest_match(0, len(chunk), 0, len(test))

# # print(pos_a, pos_b, size)
# # print(pos_a+size, pos_b+size)
# # print(len(chunk), len(test))

# # print(chunk[pos_a:pos_a+size])
# # print(test[pos_b:pos_b+size])

# # matching_ratio = len(chunk[pos_a:pos_a+size]) / len(test)
# # print(matching_ratio)
