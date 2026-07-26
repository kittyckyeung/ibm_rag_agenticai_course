import chromadb
from chromadb.utils import embedding_functions


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create a new ChromaDB client and collection
client = chromadb.Client()
collection = client.create_collection(
    name="filter_demo",
    metadata={"description": "Used to demo filtering in ChromaDB"},
    configuration={
        "embedding_function": ef
    }
    
)
print(f"Collection created: {collection.name}")
print()


# Add documents to the collection with metadata
collection.add(
    documents=[
        "This is a document about LangChain",
        "This is a reading about LlamaIndex",
        "This is a book about Python",
        "This is a document about pandas",
        "This is another document about LangChain"
    ],
    metadatas=[
        {"source": "langchain.com", "version": 0.1},
        {"source": "llamaindex.ai", "version": 0.2},
        {"source": "python.org", "version": 0.3},
        {"source": "pandas.pydata.org", "version": 0.4},
        {"source": "langchain.com", "version": 0.5},
    ],
    ids=["id1", "id2", "id3", "id4", "id5"]
)


# Retrieve documents from the collection with filtering
print("Retrieving documents from the collection with filtering:")
result=collection.get(
    where={"source": {"$eq": "langchain.com"}}
)
print(result)
print()


# Retrieve documents from the collection with filtering and version less than 0.3
print("Retrieving documents from the collection with filtering and version less than 0.3:")
result=collection.get(
    where={
        "$and": [
            {"source": {"$eq": "langchain.com"}}, 
            {"version": {"$lt": 0.3}}
        ]
    }
)
print(result)
print()


# Retrieve documents from the collection with filtering and version less than 0.3 for multiple sources
print("Retrieving documents from the collection with filtering and version less than 0.3 for multiple sources:")
result=collection.get(
    where={
        "$and": [
            {"source": {"$in": ["langchain.com", "llamaindex.ai"]}}, 
            {"version": {"$lt": 0.3}}
        ]
    }
)
print(result)
print()


# Retrieve documents from the collection with filtering for documents containing 'pandas'
print("Retrieving documents from the collection with filtering for documents containing 'pandas':")
result=collection.get(
    where_document={"$contains":"pandas"}
)
print(result)
print()


# Retrieve documents from the collection with filtering for documents containing 'LangChain' or 'Python' and version greater than 0.1
print("Retrieving documents from the collection with filtering for documents containing 'LangChain' or 'Python' and version greater than 0.1:")
result=collection.get(
    where={"version": {"$gt": 0.1}},
    where_document={
        "$or": [
            {"$contains": "LangChain"},
            {"$contains": "Python"}
        ]
    }
)
print(result)
print()

