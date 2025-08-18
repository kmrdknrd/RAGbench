import pickle
import json
import os
import asyncio
import openai
import numpy as np
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ResponseGroundedness, ResponseRelevancy
from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from tqdm import tqdm

 
def response_generation(results_list_llm_path, model_name, use_ollama = True, use_openai = False):
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
    
    if use_ollama:
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
        Answer the user's QUERY using the DOCUMENTS text.
        Keep your answer grounded in the facts of the DOCUMENTS.
        Use the IDs of the DOCUMENTS in your response.
        If the DOCUMENTS don't contain the facts to answer the QUERY, your best response is "the materials do not appear to be sufficient to provide a good answer."
        </INSTRUCTIONS>
        
        <DOCUMENTS>
        {context_texts_pretty}
        </DOCUMENTS>
        """
        
        if use_ollama:
            response = llm.invoke(rag_prompt)
        elif use_openai:
            try:
                response = openai.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": rag_prompt}],
                    max_tokens=1024,
                    n=1
                    )
                response = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error generating response for query {i}: {e}")
                response = ""
            
        results_list[i][model_name] = {
            "llm_response": response
        }
        
        if i % 10 == 0:
            pickle.dump(results_list, open(results_list_llm_path, "wb"))
        
        if i == len(results_list) - 1:
            pickle.dump(results_list, open(results_list_llm_path, "wb"))
        
    return results_list
     
async def evaluate_responses(results_list, model_name, evaluator_model_name):
    # Initialize the embedder and evaluator
    embedder = OllamaEmbeddings(model="snowflake-arctic-embed2")
    scorer = "scorer_" + evaluator_model_name
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model=evaluator_model_name))
    
    # Initialize the groundedness and relevancy scorers
    groundedness_scorer = ResponseGroundedness(llm=evaluator_llm)
    relevancy_scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=embedder)
    
    # Evaluate the LLM responses
    for i, result in tqdm(enumerate(results_list), total=len(results_list), desc="Evaluating LLM responses"):
        if scorer in result[model_name]:
            print(f"Skipping query {i} because its response has already been scored")
            continue
        
        response = result[model_name]["llm_response"]
        if "qwen" in model_name:
            response = response.split("</think>")[-1]

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
            relevancy_score = await asyncio.wait_for(
                relevancy_scorer.single_turn_ascore(sample),
                timeout=240  # 4 minutes timeout
            )
        except asyncio.TimeoutError:
            print(f"Timeout scoring query {i}'s relevancy after 4 minutes")
            relevancy_score = None
        except Exception as e:
            print(f"Error scoring query {i}'s relevancy: {e}")
            relevancy_score = None
        
        results_list[i][model_name][scorer] = {
            "groundedness_score": groundedness_score,
            "relevancy_score": relevancy_score,
            "scored": True
        }
        
        if i % 10 == 0:
            pickle.dump(results_list, open(results_list_llm_path, "wb"))
        
        if i == len(results_list) - 1: # save the results list after the last query
            pickle.dump(results_list, open(results_list_llm_path, "wb"))

## Generation and evaluation
# Process all queries in the results list
results_list_llm_path = "techqa_results/Snowflake/size2048/overlap128/mxbai-rerank-base-v2/topn8/results_llm.pkl"
model_name = "qwen3:4b"
results_list = response_generation(results_list_llm_path, model_name, use_ollama = True, use_openai = False)

# Run the async evaluation
results_list = pickle.load(open(results_list_llm_path, "rb"))
evaluator_model_name = "qwen3:0.6b"
results_list = asyncio.run(evaluate_responses(results_list, model_name, evaluator_model_name))

# Get the average scores for the groundedness and relevancy scores
scorer = "scorer_" + evaluator_model_name
groundedness_scores = np.array([result[model_name][scorer]["groundedness_score"] for result in results_list if result[model_name][scorer]["groundedness_score"] is not None])
relevancy_scores = np.array([result[model_name][scorer]["relevancy_score"] for result in results_list if result[model_name][scorer]["relevancy_score"] is not None])

print(f"Average groundedness score: {groundedness_scores.mean()}")
print(f"Average relevancy score: {relevancy_scores.mean()}")

# # Restructure the results_list dictionaries
# for result in results_list:
#     # Create the granite dictionary with the specified keys
#     # llm_results = {
#     #     'llm_response': result['llm_response'],
#     #     'groundedness_score': result['groundedness_score'],
#     #     'relevancy_score': result['relevancy_score'],
#     #     'scored': result['scored']
#     # }
    
#     # Remove the old keys
    
#     del result[model_name]
    
#     # Add the dictionary
#     result[model_name] = llm_results

# Save the restructured results
pickle.dump(results_list, open(results_list_llm_path, "wb"))

