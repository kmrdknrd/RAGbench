# RAGbench

RAGbench is a comprehensive benchmarking framework for evaluating and optimizing Retrieval-Augmented Generation (RAG) systems. This repository provides tools for testing different embedding models, chunking strategies, and reranking approaches on the TechQA dataset.

## Overview

RAGbench implements a complete RAG pipeline for benchmarking different text processing and model configurations. The framework evaluates both retrieval performance (recall) and generation quality (groundedness and relevancy).

Key components:

- **Document Processing**: Extract text from PDFs
- **Chunking and Embedding**: Test various chunk sizes (1024, 2048, 4096, 8192) and overlap strategies (0, 128)
- **Retrieval**: Benchmark different embedding models (BGE-M3, Snowflake, JINA, MXBAI)
- **Reranking**: Compare cross-encoder models for context refinement
- **Generation**: Evaluate LLM responses to the retrieved contexts
- **Evaluation**: Measure recall, groundedness, and relevance metrics

## Pipeline Architecture

1. **PDF Processing**:
   - `PdfProcessor()` extracts text from PDF documents

2. **Embedding and Retrieval**:
   - `BiEncoderPipeline()` handles text chunking and embedding
   - Retrieves semantically similar embeddings based on user queries
   - Supports various embedding models (Snowflake, BGE-M3, JINA, MXBAI)

3. **Reranking**:
   - `CrossEncoderPipeline()` refines retrieved chunks using query-document relevance
   - `FlagEmbeddingReranker()` provides alternative reranking with LLM-based models

4. **Generation and Evaluation**:
   - Generate LLM responses using the retrieved contexts
   - Evaluate response quality with groundedness and relevancy metrics

## Key Files

- `retrieval_bench.py`: Core benchmarking for retrieval pipeline components
- `generation_bench.py`: Evaluates LLM response quality based on retrieved contexts
- `new_cross_test.py`: Implementation of alternative reranking methods
- `environment.yml`: Conda environment configuration with required dependencies

## Dataset

The framework uses the TechQA dataset, which consists of technical support questions and answers. The repo includes pre-processed versions:

- `techqa.pkl`: Contains questions, expected answers, and document references
- `techqa_exp.pkl`: Expanded dataset with document texts
- `techqa_embeddings/`: Directory containing pre-computed embeddings for different models and configurations
- `techqa_results/`: Stores benchmark results for different pipeline configurations

## Usage

### Setting Up

```bash
# Clone the repository
git clone https://github.com/konradmikalauskas/RAGbench.git
cd RAGbench

# Create and activate the conda environment
conda env create -f environment.yml
conda activate ragbench
```

### Running Benchmarks

```python
# Test retrieval with different chunking strategies
python retrieval_bench.py

# Evaluate generated responses
python generation_bench.py
```

### Customizing Benchmarks

You can modify parameters in the scripts to test different configurations:

- Chunk sizes and overlap values
- Embedding models
- Reranking models and strategies
- Top-k retrieval and top-n reranking values

## Results Visualization

The repository includes tools for visualizing benchmark results:

- Recall measurement across different models and chunking strategies
- Barplots comparing model performance
- Groundedness and relevancy score analysis

## References

For more information on the retrieval and reranking pipeline approach, see:
- [UKPLab's Retrieve & Rerank Examples](https://github.com/UKPLab/sentence-transformers/tree/master/examples/sentence_transformer/applications/retrieve_rerank)

## License

See the LICENSE file for details.