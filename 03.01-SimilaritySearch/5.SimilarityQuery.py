import math
import numpy as np
import scipy
import torch
from sentence_transformers import SentenceTransformer


documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 1. embed the documents:
embeddings = model.encode(documents)
print()
print("Embeddings:", embeddings)
print()
print("Embeddings shape:", embeddings.shape)
print()

# 2. normalize the embeddings using PyTorch:
normalized_embeddings_torch = torch.nn.functional.normalize(
    torch.from_numpy(embeddings)
).numpy()
print()
print("Normalized embeddings (PyTorch):")
print(normalized_embeddings_torch)
print()

# 3. embed the query:
query_embedding = model.encode(
    ["Who is responsible for a coding project and fixing others' mistakes?"]
)
print()
print("Embeddings Query:", query_embedding)
print()
print("Embeddings Query shape:", query_embedding.shape)
print()

# 4. normalize the query embedding:
normalized_query_embedding = torch.nn.functional.normalize(
    torch.from_numpy(query_embedding)
).numpy()
print()
print("Normalized Query embeddings (PyTorch):")
print(normalized_query_embedding)
print()

# 5. calculate the cosine similarity between the documents and the query by using the dot product:
cosine_similarity_q3 = normalized_embeddings_torch @ normalized_query_embedding.T
print()
print("Cosine similarity between the documents and the query:")
print(cosine_similarity_q3)
print()

# 6. find the position of the vector with the highest cosine similarity:
highest_cossim_position = cosine_similarity_q3.argmax()
print()
print("Position of the document with the highest cosine similarity:")
print(highest_cossim_position)
print()

# 7. find the document in that position in the `documents` array:
print()
print("Document with the highest cosine similarity:")
print(documents[highest_cossim_position])
print()

# As you can see, the query retrieved the document `Bugs introduced by the intern had to be squashed by the lead developer.` which is what we would expect.
