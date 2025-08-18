from rank_bm25 import BM25Okapi
import torch
import numpy as np
import pandas as pd
import pickle
import difflib
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import openai
import requests
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from natsort import natsorted
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from FlagEmbedding import FlagLLMReranker
from mxbai_rerank import MxbaiRerankV2

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
        
    def retrieve_all(self, query, documents_embeddings):
        """Retrieve all embeddings using cosine similarity to the query"""
        
        # Embed query
        query_vector = self.model.encode([query])   
        
        # Get embeddings from documents dicts
        stored_vectors = np.array([item["vector"] for item in documents_embeddings])
        
        # Compute similarities between query and stored vectors
        similarities = cosine_similarity(query_vector, stored_vectors).flatten()
        
        return [
            {
                **documents_embeddings[i],
                "similarity": float(similarities[i])
            }
            for i in range(len(documents_embeddings))
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
            self.reranker = FlagLLMReranker(model_name, use_bf16=True)
        else:
            self.reranker = FlagLLMReranker(model_name, use_fp16=use_fp16)
    
    def rerank(self, query, documents, top_n=4):
        """Rerank documents using FlagLLMReranker"""
        # Extract texts and create query-passage pairs
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Compute scores for all pairs
        scores = self.reranker.compute_score(pairs)
        
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

def retrieve_and_rerank(queries, embeddings, bi_encoder, cross_encoder, dataset, top_k=50, top_n=4, hyde_mode = False, hybrid_search = False, bm25_weight = 0.1, save_results=False, save_path=None):
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
        
    if hybrid_search:
        if Path("data/tokenized_techqa.pkl").exists():
            print("Loading tokenized corpus from pickle file")
            with open("data/tokenized_techqa.pkl", "rb") as f:
                tokenized_corpus = pickle.load(f)
        else:
            corpus = [doc["text"] for doc in embeddings]
            tokenized_corpus = [nltk.word_tokenize(doc) for doc in corpus]
            with open("data/tokenized_techqa.pkl", "wb") as f:
                pickle.dump(tokenized_corpus, f)
        
        bm25 = BM25Okapi(tokenized_corpus)   
    
    results_list = []
    for i, query in tqdm(enumerate(queries), total=len(queries), desc="Processing queries"):        
        # Generate hypothetical document using OpenAI API
        if hyde_mode:
            hyde_query_prompt = f"""Write a document that answers the following question:\nQuestion: {query}\nDocument:"""
            # Generate hypothetical document using OpenAI API
            try:
                response = openai.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": hyde_query_prompt}],
                    max_tokens=512,
                    n=1
                )
                hypothetical_doc = response.choices[0].message.content.strip()
                # Use the hypothetical document as the query for retrieval
                hyde_query = hypothetical_doc
                
                top_hyde_chunks = bi_encoder.retrieve_top_k(hyde_query, embeddings, top_k=top_k)  # Retrieve top chunks for each query
                reranked_hyde_results = cross_encoder.rerank(query, top_hyde_chunks, top_n=top_n)  # Rerank the chunks
                reranked_hyde_results = [{k: v for k, v in d.items() if k not in ["vector", "match_types"]} for d in reranked_hyde_results]
                
            except Exception as e:
                print(f"Error generating hypothetical document for query {i}: {e}")
                # Fall back to original query if API call fails
                pass
            
        if hybrid_search:
            all_chunks_dense = bi_encoder.retrieve_all(query, embeddings)
            all_chunks_dense_scores = np.array([d["similarity"] for d in all_chunks_dense])
            all_chunks_bm25_scores = bm25.get_scores(nltk.word_tokenize(query))
            all_chunks_bm25_scores = minmax_scale(all_chunks_bm25_scores) # normalize bm25 scores (0-1)
            
            # # histogram of all_chunks_dense_scores
            # plt.hist(all_chunks_dense_scores, bins=100)
            # plt.show()
            
            # # histogram of all_chunks_bm25_scores
            # plt.hist(all_chunks_bm25_scores, bins=100)
            # plt.show()
            
            all_chunks_scores = all_chunks_dense_scores + all_chunks_bm25_scores * bm25_weight # combine scores
            
            # get top k chunks            
            top_chunks_indices = np.argsort(all_chunks_scores)[-top_k:][::-1]
            top_chunks = [
                {
                    **all_chunks_dense[j],
                    "bm25_score": all_chunks_bm25_scores[j],
                    "combined_score": all_chunks_scores[j]
                }
                for j in top_chunks_indices
            ]     
        else:   
            top_chunks = bi_encoder.retrieve_top_k(query, embeddings, top_k=top_k)  # Retrieve top chunks for each query
        
        reranked_results = cross_encoder.rerank(query, top_chunks, top_n=top_n)  # Rerank the chunks
        reranked_results = [{k: v for k, v in d.items() if k not in ["vector", "match_types"]} for d in reranked_results]
        
        expected_sentences = dataset.loc[i, "all_relevant_sentence_keys"]
        
        full_results = {"query": query,
                        "question_id": dataset.loc[i, "question_id"],
                        "expected": expected_sentences,
                        "results": reranked_results}
        if hyde_mode:
            full_results["hyde_results"] = reranked_hyde_results
            
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

def prepare_techqa():
    """
    Prepare the TechQA dataset for retrieval benchmarking.
    
    This function loads the TechQA dataset, filters it, processes documents,
    removes duplicates, and saves the processed data.
    
    Returns:
        tuple: (techqa, techqa_exp) - The processed TechQA dataset and expanded version
    """
    # Check if techqa.pkl exists
    if Path("techqa.pkl").exists():
        print("Loading techqa from pickle file")
        with open("techqa.pkl", "rb") as f:
            techqa = pickle.load(f)
    else:
        # Prepare TechQA dataset
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

        # Save techqa
        with open("techqa.pkl", "wb") as f:
            pickle.dump(techqa, f)

    # Check if techqa_exp.pkl exists
    if Path("techqa_exp.pkl").exists():
        print("Loading techqa_exp from pickle file")
        with open("techqa_exp.pkl", "rb") as f:
            techqa_exp = pickle.load(f)
    else:
        ##### GET DOCUMENTS, REMOVE DUPLICATES #####
        # Create a new dataframe with each document as a separate row
        techqa_exp = techqa.explode(list(('documents', 'documents_sentences'))).reset_index(drop=True)

        # Create a new 'doc_id' column that combines question_id with document number
        techqa_exp['doc_id'] = techqa_exp.groupby('question_id').cumcount() + 1
        techqa_exp['doc_id'] = techqa_exp['question_id'].apply(lambda x: f'{x}') + '-' + techqa_exp['doc_id'].apply(lambda x: f'doc{x}')

        # Keep only the 'documents', 'doc_id', and 'documents_sentences' columns
        techqa_exp = techqa_exp[["documents", "doc_id", "documents_sentences"]]

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
        techqa_exp = techqa_exp.reset_index(drop=True)

        # Save techqa_exp
        with open("techqa_exp.pkl", "wb") as f:
            pickle.dump(techqa_exp, f)
        
    return techqa, techqa_exp    

def compute_recall(results_list, hyde_mode = False):
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
        if hyde_mode:
            results_sentences = [match for matches in result["hyde_results"] for match in [matches["sentence_matches"]]]
            results_doc_ids = [id for ids in result["hyde_results"] for id in [ids["original_doc_id"]]]
        else:
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

# ##### TECHQA EMBEDDING #####
techqa, techqa_exp = prepare_techqa()
techqa_questions = techqa.question.tolist()

bi_encoder_model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
cross_encoder = CrossEncoderPipeline(model_name=cross_encoder_model_name)

# flag_reranker = FlagEmbeddingReranker(model_name="BAAI/bge-reranker-v2-gemma")

# mxbai_reranker = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-base-v2")

bi_encoder_model_name_short = bi_encoder_model_name.split("/")[0]
cross_encoder_model_name_short = "miniLM-L6-v2"

c_size = 2048
c_overlap = 128
bi_encoder_text_embedding = BiEncoderPipeline(
    model_name=bi_encoder_model_name,
    chunk_size=c_size,
    chunk_overlap=c_overlap
    )
    
# Load embeddings
with open(f"techqa_embeddings/{bi_encoder_model_name_short}/size{c_size}/overlap{c_overlap}/embeddings_matched.pkl", "rb") as f:
    techqa_embed_final = pickle.load(f)

bm25_weights = np.arange(0, 1.55, 0.05)

for bm25_weight in bm25_weights:
    results_bm25 = retrieve_and_rerank(queries=techqa_questions, 
                                       embeddings=techqa_embed_final, 
                                       bi_encoder=bi_encoder_text_embedding, 
                                       cross_encoder=cross_encoder, 
                                       dataset=techqa, 
                                       top_k=50, 
                                       top_n=8,
                                       hyde_mode=False,
                                       hybrid_search=True,
                                       bm25_weight=bm25_weight)

    # Save results
    with open(f"tests/hybrid_search/bm25_weight_{bm25_weight}.pkl", "wb") as f:
        pickle.dump(results_bm25, f)

# with open(f"tests/hybrid_search/bm25_weight_0.5.pkl", "rb") as f:
#     results_bm25 = pickle.load(f)

results_dir = Path("tests/hybrid_search")
# Create line plot of recall vs bm25 weight
recall_means = []
for f in natsorted(results_dir.glob("*.pkl")):
    with open(f, "rb") as f:
        results_bm25 = pickle.load(f)
    recall_mean, recalls = compute_recall(results_bm25, hyde_mode=False)
    recall_means.append(recall_mean * 100)

plt.figure(figsize=(10, 6))
plt.plot(bm25_weights, recall_means, marker='o', linestyle='-', linewidth=2)
plt.title('Recall vs. BM25 Weight')
plt.xlabel('BM25 Weight')
plt.ylabel('Recall (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, 1.6, 0.1))
plt.ylim(59.2, 60.2)
plt.tight_layout()
plt.show()