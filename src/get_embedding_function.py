# Using a local embedding model for now. However, this is likely the biggest bottleneck regarding model performance.
# Ideally switch to a paid service for embedding model. 
from langchain_ollama import OllamaEmbeddings

# tutorial used AWS bedrock instead 

def get_embedding_function():
    embeddings = OllamaEmbeddings(model = "bge-m3")
    return embeddings 