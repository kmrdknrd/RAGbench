import pickle
import json
import os
import asyncio
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ResponseGroundedness, ResponseRelevancy
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import numpy as np

def response_generation(results_list_llm_path, model_name, first="instructions", second="context", context_asc=False):
    # Check if the first and second arguments are valid and not the same
    if first not in ["instructions", "context", "query"]:
        raise ValueError(f"Invalid first argument: {first}")
    if second not in ["instructions", "context", "query"]:
        raise ValueError(f"Invalid second argument: {second}")
    if first == second:
        raise ValueError(f"First and second arguments cannot be the same: {first}")
    
    # Check if results list LLM file exists
    if os.path.exists(results_list_llm_path):
        results_list = pickle.load(open(results_list_llm_path, "rb"))
        results_list_llm_exists = True
        print(f"Loading results list from {results_list_llm_path}")
    else:
        results_list_no_llm_path = results_list_llm_path.replace("llm.pkl", ".pkl")
        results_list = pickle.load(open(results_list_no_llm_path, "rb"))
        results_list_llm_exists = False
        print(f"Results list LLM file does not exist, creating new file {results_list_llm_path}")
    
    llm = OllamaLLM(model=model_name)
    for i, result in tqdm(enumerate(results_list), total=len(results_list), desc="Generating LLM responses to queries + retrieved contexts"):
        # Skip if already has an LLM response
        if results_list_llm_exists:
            if model_name in result:
                if "llm_response" in result[model_name]:
                    print(f"Skipping query {i} because it already has an LLM response")
                    continue
        
        query = result["query"]
        context_ids = [entry["original_doc_id"] for entry in result["results"]]
        context_texts = [entry["text"] for entry in result["results"]]
        context_texts = context_texts[::-1] if context_asc else context_texts # reverse the order of the context texts
        context_texts_pretty = "\n".join([f"<DOCUMENT{i+1}>\nID: {context_ids[i]}\nTEXT:\n{text}\n</DOCUMENT{i+1}>\n" for i, text in enumerate(context_texts)])
        
        instructions = f"""
        <INSTRUCTIONS>
        Answer the user's QUERY using the text in DOCUMENTS.
        Keep your answer grounded in the facts of the DOCUMENTS.
        If the DOCUMENTS don't contain the facts to answer the QUERY, your best response is "the materials do not appear to be sufficient to provide a good answer."
        </INSTRUCTIONS>
        """
        
        context = f"""
        <DOCUMENTS>
        {context_texts_pretty}
        </DOCUMENTS>
        """
        
        query = f"""
        <QUERY>
        {query}
        </QUERY>
        """
        
        if first == "instructions":
            if second == "context":
                rag_prompt = f"""
                {instructions}
                {context}
                {query}
                """
            elif second == "query":
                rag_prompt = f"""
                {instructions}
                {query}
                {context}
                """
        elif first == "context":
            if second == "instructions":
                rag_prompt = f"""
                {context}
                {instructions}
                {query}
                """
            elif second == "query":
                rag_prompt = f"""
                {context}
                {query}
                {instructions}
                """
        else: # first == "query"
            if second == "instructions":
                rag_prompt = f"""
                {query}
                {instructions}
                {context}
                """
            elif second == "context":
                rag_prompt = f"""
                {query}
                {context}
                {instructions}
                """
    
        
        response = llm.invoke(rag_prompt)
        results_list[i][model_name] = {
            "llm_response": response
        }
        
        third = "context_asc" if context_asc else "context_desc"
        output_path = f"tests/order/{first}/{second}/{third}/results.pkl"
        
        if i % 10 == 0:
            pickle.dump(results_list, open(output_path, "wb"))
        
        if i == len(results_list) - 1:
            pickle.dump(results_list, open(output_path, "wb"))
        
    return results_list
     
async def evaluate_responses(results_list, results_list_path, model_name):
    # Initialize the embedder and evaluator
    embedder = OllamaEmbeddings(model="snowflake-arctic-embed2")
    evaluator_model_name = "qwen3:0.6b"
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model=evaluator_model_name))
    
    # Initialize the groundedness and relevancy scorers
    groundedness_scorer = ResponseGroundedness(llm=evaluator_llm)
    relevancy_scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=embedder)
    
    # Evaluate the LLM responses
    for i, result in tqdm(enumerate(results_list), total=len(results_list), desc="Evaluating LLM responses"):
        if "scored" in result[model_name]:
            print(f"Skipping query {i} because its response has already been scored")
            continue
        
        response = result[model_name]["llm_response"]

        sample = SingleTurnSample(
            user_input=result["query"],
            response=response,
            retrieved_contexts=[entry["text"] for entry in result["results"]]   
        )

        try:
            groundedness_score = await groundedness_scorer.single_turn_ascore(sample)
        except Exception as e:
            print(f"Error scoring query {i}'s groundedness: {e}")
            groundedness_score = None
        
        try:
            relevancy_score = await relevancy_scorer.single_turn_ascore(sample)
        except Exception as e:
            print(f"Error scoring query {i}'s relevancy: {e}")
            relevancy_score = None
        
        results_list[i][model_name]["groundedness_score"] = groundedness_score
        results_list[i][model_name]["relevancy_score"] = relevancy_score
        results_list[i][model_name]["scored"] = True
        
        results_list_path = results_list_path.replace("results.pkl", "results_scored.pkl")
        
        if i % 10 == 0:
            pickle.dump(results_list, open(results_list_path, "wb"))
        
        if i == len(results_list) - 1: # save the results list after the last query
            pickle.dump(results_list, open(results_list_path, "wb"))

## Generation and evaluation
# Process all queries in the results list
# results_list_test_path = "tests/order/snowflake_2048_128_8.pkl"
# results_list = pickle.load(open(results_list_test_path, "rb"))

# random.seed(42)
# results_list_sample = random.sample(results_list, 100)

results_list_sample_path = "tests/order/snowflake_2048_128_8_sample.pkl"
# pickle.dump(results_list_sample, open(results_list_sample_path, "wb"))
results_list_sample = pickle.load(open(results_list_sample_path, "rb"))

context_conditions = [True, False]
order_conditions = ["instructions", "context", "query"]
all_order_conditions = [
    (order_conditions[0], order_conditions[1], order_conditions[2]),
    (order_conditions[0], order_conditions[2], order_conditions[1]),
    (order_conditions[1], order_conditions[0], order_conditions[2]),
    (order_conditions[1], order_conditions[2], order_conditions[0]),
    (order_conditions[2], order_conditions[0], order_conditions[1]),
    (order_conditions[2], order_conditions[1], order_conditions[0])
]

model_name = "cogito:3b"
# for order_condition in all_order_conditions:
#     for context_condition in context_conditions:
#         print(f"Order condition: {order_condition}, Context ascending: {context_condition}")
#         results_list = response_generation(results_list_sample_path, model_name, first=order_condition[0], second=order_condition[1], context_asc=context_condition)

# for order_condition in all_order_conditions:
#     for context_condition in context_conditions:
#         print(f"Order condition: {order_condition}, Context ascending: {context_condition}")
#         context_condition_path = "context_asc" if context_condition else "context_desc"
        
#         results_list_path = f"tests/order/{order_condition[0]}/{order_condition[1]}/{context_condition_path}/results.pkl"
#         results_list = pickle.load(open(results_list_path, "rb"))
#         asyncio.run(evaluate_responses(results_list, results_list_path, model_name))


## Plotting    
# Create bar plot of the instructions, context and query combination (i.e., all_order_conditions[0]) (both ascending and descending bars)
def plot_condition_results(condition, model_name, y_lim_groundedness=None, y_lim_relevancy=None):
    """
    Create bar plots comparing ascending and descending context order for a given condition.
    
    Args:
        condition: Tuple of (first, second, third) order elements
        model_name: Name of the model used for evaluation
    
    Returns:
        fig: The matplotlib figure object containing the plots
    """
    results_asc_path = Path(f"tests/order/{condition[0]}/{condition[1]}/context_asc/results_scored.pkl")
    results_desc_path = Path(f"tests/order/{condition[0]}/{condition[1]}/context_desc/results_scored.pkl")

    results_asc = pickle.load(open(results_asc_path, "rb"))
    results_desc = pickle.load(open(results_desc_path, "rb"))

    # Get the groundedness and relevancy scores for the ascending and descending results (ignore None and nan values)
    groundedness_asc_mean = np.mean([result[model_name]["groundedness_score"] for result in results_asc if result[model_name]["groundedness_score"] is not None and not np.isnan(result[model_name]["groundedness_score"])])
    groundedness_desc_mean = np.mean([result[model_name]["groundedness_score"] for result in results_desc if result[model_name]["groundedness_score"] is not None and not np.isnan(result[model_name]["groundedness_score"])])
    relevancy_asc_mean = np.mean([result[model_name]["relevancy_score"] for result in results_asc if result[model_name]["relevancy_score"] is not None and not np.isnan(result[model_name]["relevancy_score"])])
    relevancy_desc_mean = np.mean([result[model_name]["relevancy_score"] for result in results_desc if result[model_name]["relevancy_score"] is not None and not np.isnan(result[model_name]["relevancy_score"])])

    # Create a bar plot of the groundedness and relevancy scores for the ascending and descending results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Create DataFrames for each metric
    groundedness_data = pd.DataFrame({
        'Order': ['Ascending', 'Descending'],
        'Score': [groundedness_asc_mean, groundedness_desc_mean]
    })

    relevancy_data = pd.DataFrame({
        'Order': ['Ascending', 'Descending'],
        'Score': [relevancy_asc_mean, relevancy_desc_mean]
    })

    # Plot groundedness
    sns.barplot(x='Order', y='Score', data=groundedness_data, ax=ax1, palette="viridis")
    ax1.set_xlabel("Order")
    ax1.set_ylabel("Score")
    ax1.set_title("Groundedness")

    # Plot relevancy
    sns.barplot(x='Order', y='Score', data=relevancy_data, ax=ax2, palette="viridis")
    ax2.set_xlabel("Order")
    ax2.set_ylabel("Score")
    ax2.set_title("Relevancy")
    
    if y_lim_groundedness:
        ax1.set_ylim(0, y_lim_groundedness)
    if y_lim_relevancy:
        ax2.set_ylim(0, y_lim_relevancy)

    # Add a title for the whole plot
    fig.suptitle(f"{condition[0].capitalize()} > {condition[1].capitalize()} > {condition[2].capitalize()}", fontsize=16)
    plt.tight_layout()
    
    return fig

condition = all_order_conditions[0]
model_name = "cogito:3b"

all_figures = []

# find max y-value for all figures
maxgroundedness = 0
maxrelevancy = 0
for condition in all_order_conditions:
    results_asc_path = Path(f"tests/order/{condition[0]}/{condition[1]}/context_asc/results_scored.pkl")
    results_desc_path = Path(f"tests/order/{condition[0]}/{condition[1]}/context_desc/results_scored.pkl")
    
    results_asc = pickle.load(open(results_asc_path, "rb"))
    results_desc = pickle.load(open(results_desc_path, "rb"))
    
    # Get the groundedness and relevancy scores for the ascending and descending results (ignore None and nan values)
    groundedness_asc_mean = np.mean([result[model_name]["groundedness_score"] for result in results_asc if result[model_name]["groundedness_score"] is not None and not np.isnan(result[model_name]["groundedness_score"])])
    groundedness_desc_mean = np.mean([result[model_name]["groundedness_score"] for result in results_desc if result[model_name]["groundedness_score"] is not None and not np.isnan(result[model_name]["groundedness_score"])])
    relevancy_asc_mean = np.mean([result[model_name]["relevancy_score"] for result in results_asc if result[model_name]["relevancy_score"] is not None and not np.isnan(result[model_name]["relevancy_score"])])
    relevancy_desc_mean = np.mean([result[model_name]["relevancy_score"] for result in results_desc if result[model_name]["relevancy_score"] is not None and not np.isnan(result[model_name]["relevancy_score"])])
    
    if groundedness_asc_mean > maxgroundedness:
        maxgroundedness = groundedness_asc_mean
    if groundedness_desc_mean > maxgroundedness:
        maxgroundedness = groundedness_desc_mean
    if relevancy_asc_mean > maxrelevancy:
        maxrelevancy = relevancy_asc_mean   
    if relevancy_desc_mean > maxrelevancy:
        maxrelevancy = relevancy_desc_mean
        

for condition in all_order_conditions:
    fig = plot_condition_results(condition, model_name, y_lim_groundedness=maxgroundedness * 1.1, y_lim_relevancy=maxrelevancy * 1.1)
    all_figures.append(fig)



