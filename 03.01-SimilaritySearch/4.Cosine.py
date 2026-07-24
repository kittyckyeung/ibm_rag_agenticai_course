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
embeddings = model.encode(documents)

print()
print("Embeddings:", embeddings)
print()
print("Embeddings shape:", embeddings.shape)
print()

# L2 norms
l2_norms = np.sqrt(np.sum(embeddings**2, axis=1))
print()
print("L2 norms:")
print(l2_norms)
print()


# L2 norms reshaped
l2_norms_reshaped = l2_norms.reshape(-1,1)
print()
print("L2 norms reshaped:")
print(l2_norms_reshaped)
print()

# Normalized embedding vectors by dividing every component in the vector by the vector's L2 norm
normalized_embeddings_manual = embeddings/l2_norms_reshaped
print()
print("Normalized embeddings (manual):")
print(normalized_embeddings_manual)
print()


# Verify that normalized_embeddings_manual are normalized vectors by making sure that the length of each vector is equal to 1.
print()
print("Verification (should be all ones):")
print(np.sqrt(np.sum(normalized_embeddings_manual**2, axis=1)))
print()


# Normalize embeddings using PyTorch
normalized_embeddings_torch = torch.nn.functional.normalize(
    torch.from_numpy(embeddings)
).numpy()
print()
print("Normalized embeddings (PyTorch):")
print(normalized_embeddings_torch)
print()

print()
print("Verify that the normalized embeddings we calculated manually and the normalized embeddings calculated using torch are close:")
print(np.allclose(normalized_embeddings_manual, normalized_embeddings_torch))
print()


def dot_product_fn(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))


# Calculates the cosine similarity between the vector embeddings of the first and second sentence
print()
print("Dot product between first and second embedding:")
print(dot_product_fn(normalized_embeddings_manual[0], normalized_embeddings_manual[1]))
print()

cosine_similarity_manual = np.empty([4,4])
for i in range(normalized_embeddings_manual.shape[0]):
    for j in range(normalized_embeddings_manual.shape[0]):
        cosine_similarity_manual[i,j] = dot_product_fn(
            normalized_embeddings_manual[i], 
            normalized_embeddings_manual[j]
        )

print()
print(f"Manual Cosine Similarity matrix:")
print(cosine_similarity_manual)
print()


# Calculate cosine similarity using matrix multiplication
cosine_similarity_operator = normalized_embeddings_manual @ normalized_embeddings_manual.T

print()
print("Cosine Similarity matrix using the matrix multiplication operator:")
print(cosine_similarity_operator)
print()


print()
print("Verifying that the manual Cosine Similarity matrix is equal to the matrix multiplication operator result")
print(np.allclose(cosine_similarity_manual, cosine_similarity_operator))
print()
