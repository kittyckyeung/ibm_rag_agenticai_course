# Setup
import chromadb
from chromadb.utils import embedding_functions


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# Collection creation
client = chromadb.Client()
collection = client.create_collection(
    name="my_collection_name",
    metadata={"topic": "query testing"},
    configuration={
        "hnsw": {
            "space": "cosine",
            "ef_search": 100,
            "ef_construction": 100,
            "max_neighbors": 16
        },
        "embedding_function": ef
    }
)


print(f"Collection created: {collection.name}")
print()


collection.add(
    documents=[
        "Giant pandas are a bear species that lives in mountainous areas.",
        "A pandas DataFrame stores two-dimensional, tabular data",
        "I think everyone agrees that pandas are some of the cutest animals on the planet",
        "A direct comparison between pandas and polars indicates that polars is a more efficient library than pandas.",
    ],
    metadatas=[
        {"topic": "animals"},
        {"topic": "data analysis"},
        {"topic": "animals"},
        {"topic": "data analysis"},
    ],
    ids=["id1", "id2", "id3", "id4"]
)


# Querying the collection
print("Query results for 'cats':")
result=collection.query(
    query_texts=["cats"],
    n_results=10,
)
print(result)
print()


# Querying the collection
print("Query results for 'polar bear':")
result=collection.query(
    query_texts=["polar bear"],
    n_results=1,
)
print(result)
print()


# Querying the collection with a filter
print("Query results for 'polar bear' with topic 'animals':")
result=collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where={'topic': 'animals'}
)
print(result)
print()


# Querying the collection with a filter
print("Query results for 'polar bear' excluding documents containing 'library':")
result=collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where_document={'$not_contains': 'library'}
)
print(result)
print()


# Querying the collection with a filter
print("Query results for 'polar bear' with topic 'animals' and excluding documents containing 'library':")
result=collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where={'topic': 'animals'},
    where_document={'$not_contains': 'library'}
)
print(result)
print()

