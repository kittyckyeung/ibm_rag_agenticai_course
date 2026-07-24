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

def dot_product_fn(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))

print()
print("Dot product between first and second embedding:")
print(dot_product_fn(embeddings[0], embeddings[1]))
print("Dot product between second and first embedding:")
print(dot_product_fn(embeddings[1], embeddings[0]))
print()

dot_product_manual = np.empty([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        dot_product_manual[i,j] = dot_product_fn(embeddings[i], embeddings[j])

print()
print(f"Manual Dot Product matrix:")
print(dot_product_manual)
print()
print("Manual Dot Product between first and second embedding:", dot_product_manual[0,1])
print()
print("Manual Dot Product between second and first embedding:", dot_product_manual[1,0])
print()

# Matrix multiplication operator
dot_product_operator = embeddings @ embeddings.T
print()
print("Dot Product matrix using the matrix multiplication operator:")
print(dot_product_operator)
print()
print("Verifying that the manual Dot Product matrix is equal to the matrix multiplication operator result")
print(np.allclose(dot_product_manual, dot_product_operator, atol=1e-05))
print()
print("Dot Product matrix using np.matmul:")
print(np.matmul(embeddings,embeddings.T))
print()
print("Dot Product matrix using np.dot:")
print(np.dot(embeddings,embeddings.T))
print()

dot_product_distance = -dot_product_manual
print()
print("Dot Product distance matrix:")
print(dot_product_distance)
print()

