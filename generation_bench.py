import pickle
import json
import os
import asyncio
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ResponseGroundedness, ResponseRelevancy
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import numpy as np
 
def response_generation(results_list_llm_path, model_name):
    # check if results list LLM file exists
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
        context_texts_pretty = "\n".join([f"<DOCUMENT{i+1}: {context_ids[i]}>\nTEXT:\n{text}\n</DOCUMENT{i+1}: {context_ids[i]}>\n" for i, text in enumerate(context_texts)])
        
        rag_prompt = f"""        
        <QUERY>
        {query}
        </QUERY>
        
        <INSTRUCTIONS>
        Answer the user's QUERY using the DOCUMENTS text above.
        Keep your answer grounded in the facts of the DOCUMENTS.
        Use the IDs of the DOCUMENTS in your response.
        If the DOCUMENTS don't contain the facts to answer the QUERY, your best response is "the materials do not appear to be sufficient to provide a good answer."
        </INSTRUCTIONS>
        
        <DOCUMENTS>
        {context_texts_pretty}
        </DOCUMENTS>
        """
        
        response = llm.invoke(rag_prompt)
        results_list[i][model_name] = {
            "llm_response": response
        }
        
        if i % 10 == 0:
            pickle.dump(results_list, open(results_list_llm_path, "wb"))
        
        if i == len(results_list) - 1:
            pickle.dump(results_list, open(results_list_llm_path, "wb"))
        
    return results_list
     
async def evaluate_responses(results_list, model_name):
    # Initialize the embedder and evaluator
    embedder = OllamaEmbeddings(model="snowflake-arctic-embed2")
    evaluator_model_name = "gemma3:1b"
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
        
        if i % 10 == 0:
            pickle.dump(results_list, open(results_list_llm_path, "wb"))
        
        if i == len(results_list) - 1: # save the results list after the last query
            pickle.dump(results_list, open(results_list_llm_path, "wb"))

## Generation and evaluation
# Process all queries in the results list
results_list_llm_path = "techqa_results/Snowflake/size4096/overlap128/miniLM-L6-v2/topn4/results_llm.pkl"
model_name = "qwen3:1.7b"
results_list = response_generation(results_list_llm_path, model_name)

# Run the async evaluation
results_list = pickle.load(open(results_list_llm_path, "rb"))
asyncio.run(evaluate_responses(results_list, model_name))

# Get the average scores for the groundedness and relevancy scores
groundedness_scores = np.array([result[model_name]["groundedness_score"] for result in results_list if result[model_name]["groundedness_score"] is not None])
relevancy_scores = np.array([result[model_name]["relevancy_score"] for result in results_list if result[model_name]["relevancy_score"] is not None])

print(f"Average groundedness score: {groundedness_scores.mean()}")
print(f"Average relevancy score: {relevancy_scores.mean()}")

# # Restructure the results_list dictionaries
# for result in results_list:
#     # Create the granite dictionary with the specified keys
#     llm_results = {
#         'llm_response': result['llm_response'],
#         'groundedness_score': result['groundedness_score'],
#         'relevancy_score': result['relevancy_score'],
#         'scored': result['scored']
#     }
    
#     # Remove the old keys
#     del result['llm_response']
#     del result['groundedness_score']
#     del result['relevancy_score']
#     del result['scored']
    
#     # Add the dictionary
#     result[model_name] = llm_results

# # Save the restructured results
# pickle.dump(results_list, open(results_list_llm_path, "wb"))


# Get the average scores for the groundedness and relevancy scores (skip index 231)
groundedness_scores = np.array([result["cogito:3b"]["groundedness_score"] for result in results_list if result["cogito:3b"]["groundedness_score"] is not None])
relevancy_scores = np.array([result["cogito:3b"]["relevancy_score"] for result in results_list if result["cogito:3b"]["relevancy_score"] is not None])

print(f"Average groundedness score: {groundedness_scores.mean()}")
print(f"Average relevancy score: {relevancy_scores.mean()}")

for i, result in enumerate(results_list):
    print(result["cogito:3b"]["groundedness_score"])



# Get the average scores for the groundedness and relevancy scores
groundedness_scores = np.array([result["cogito:8b"]["groundedness_score"] for result in results_list if result["cogito:8b"]["groundedness_score"] is not None])
relevancy_scores = np.array([result["cogito:8b"]["relevancy_score"] for result in results_list if result["cogito:8b"]["relevancy_score"] is not None])

print(f"Average groundedness score: {groundedness_scores.mean()}")
print(f"Average relevancy score: {relevancy_scores.mean()}")




# Get the average scores for the groundedness and relevancy scores
groundedness_scores = np.array([result["llama3.2:latest"]["groundedness_score"] for result in results_list if result["llama3.2:latest"]["groundedness_score"] is not None])
relevancy_scores = np.array([result["llama3.2:latest"]["relevancy_score"] for result in results_list if result["llama3.2:latest"]["relevancy_score"] is not None])

print(f"Average groundedness score: {groundedness_scores.mean()}")
print(f"Average relevancy score: {relevancy_scores.mean()}")