# RAGbench
RAGbench.py emulates the AnythingLLM RAG pipeline

- PdfProcessor() reads .pdf's into Python
- BiEncoderPipeline() sets up a bi-encoder for text chunking and embedding, and – given a user query – retrieving semantically similar embeddings
- CrossEncoderPipeline() sets up a cross-encoder for reranking the retrieved embeddings based on the query

For more information on the retrieval and reranking pipeline, see: https://github.com/UKPLab/sentence-transformers/tree/master/examples/sentence_transformer/applications/retrieve_rerank
