import pickle
import json
import os
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ResponseGroundedness, ResponseRelevancy
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import numpy as np

# Process all queries in the results list
model_name = "cogito:3b"
llm = OllamaLLM(model=model_name)

# check if results list LLM file exists
results_list_llm = "techqa_results/Snowflake/results_list_size-4096_overlap-128_llm.pkl"
if os.path.exists(results_list_llm):
    results_list = pickle.load(open(results_list_llm, "rb"))
    results_list_llm_exists = True
    print(f"Loading results list from {results_list_llm}")
else:
    results_list = pickle.load(open("techqa_results/Snowflake/results_list_size-4096_overlap-128.pkl", "rb"))
    results_list_llm_exists = False
    print(f"Results list LLM file does not exist, creating new file {results_list_llm}")

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
    context_texts_pretty = "\n".join([f"<DOCUMENT{i+1}>\nID: {context_ids[i]}\nTEXT:\n{text}\n</DOCUMENT{i+1}>\n" for i, text in enumerate(context_texts)])

    rag_prompt = f"""
    <DOCUMENTS>
    {context_texts_pretty}
    </DOCUMENTS>
    
    <QUERY>
    {query}
    </QUERY>
    
    <INSTRUCTIONS>
    Answer the user's QUERY using the DOCUMENTS text above.
    Keep your answer grounded in the facts of the DOCUMENTS.
    If the DOCUMENTS don't contain the facts to answer the QUERY, your best response is "the materials do not appear to be sufficient to provide a good answer."
    </INSTRUCTIONS>
    """

    response = llm.invoke(rag_prompt)
    results_list[i][model_name] = {
        "llm_response": response
    }
    
    pickle.dump(results_list, open(f"techqa_results/Snowflake/results_list_size-4096_overlap-128_llm.pkl", "wb"))
    
# Load results list from file
results_list = pickle.load(open(f"techqa_results/Snowflake/results_list_size-4096_overlap-128_llm.pkl", "rb"))

evaluator_llm = LangchainLLMWrapper(ChatOllama(model=model_name))
embed = OllamaEmbeddings(model="nomic-embed-text")
groundedness_scorer = ResponseGroundedness(llm=evaluator_llm)
relevancy_scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=embed)
for i, result in tqdm(enumerate(results_list), total=len(results_list), desc="Evaluating LLM responses"):
    if "scored" in result[model_name]:
        print(f"Skipping query {i} because its response has already been scored")
        continue
    
    response = result[model_name]["llm_response"]
    # response = response.split("</think>")[1].strip()

    sample = SingleTurnSample(
        user_input=result["query"],
        response=response,
        retrieved_contexts=[entry["text"] for entry in result["results"]]   
    )

    groundedness_score = await groundedness_scorer.single_turn_ascore(sample)
    relevancy_score = await relevancy_scorer.single_turn_ascore(sample)
    
    results_list[i][model_name]["groundedness_score"] = groundedness_score
    results_list[i][model_name]["relevancy_score"] = relevancy_score
    results_list[i][model_name]["scored"] = True
    
    if i % 10 == 0:
        pickle.dump(results_list, open(f"techqa_results/Snowflake/results_list_size-4096_overlap-128_llm.pkl", "wb"))

# Get the average scores for the groundedness and relevancy scores
groundedness_scores = np.array([result[model_name]["groundedness_score"] for result in results_list])
relevancy_scores = np.array([result[model_name]["relevancy_score"] for result in results_list])

print(f"Average groundedness score: {groundedness_scores.mean()}")
print(f"Average relevancy score: {relevancy_scores.mean()}")

# Restructure the results_list dictionaries
for result in results_list:
    # Create the granite dictionary with the specified keys
    llm_results = {
        'llm_response': result['llm_response'],
        'groundedness_score': result['groundedness_score'],
        'relevancy_score': result['relevancy_score'],
        'scored': result['scored']
    }
    
    # Remove the old keys
    del result['llm_response']
    del result['groundedness_score']
    del result['relevancy_score']
    del result['scored']
    
    # Add the granite dictionary
    result[model_name] = llm_results

# Save the restructured results
pickle.dump(results_list, open(f"techqa_results/Snowflake/results_list_size-4096_overlap-128_llm.pkl", "wb"))


