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

def euclidean_distance_fn(vector1, vector2):
    squared_sum = sum((x - y) ** 2 for x, y in zip(vector1, vector2))
    return math.sqrt(squared_sum)

print()
print("Euclidean distance between first and second embedding:", euclidean_distance_fn(embeddings[0], embeddings[1]))
print()
print("Euclidean distance between second and first embedding:", euclidean_distance_fn(embeddings[1], embeddings[0]))
print()

l2_dist_manual = np.zeros([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        ## Less efficient way to calculate the L2 distance matrix
        ##l2_dist_manual[i,j] = euclidean_distance_fn(embeddings[i], embeddings[j])
        ## More efficient way to calculate the L2 distance matrix
        if j > i: # Calculate the upper triangle only
            l2_dist_manual[i,j] = euclidean_distance_fn(embeddings[i], embeddings[j])
        elif i > j: # Copy the uper triangle to the lower triangle
            l2_dist_manual[i,j] = l2_dist_manual[j,i]

print()
print(f"Manual L2 distance matrix:")
print(l2_dist_manual)
print()
print("Manual L2 distance between first and second embedding:", l2_dist_manual[0,1])
print()
print("Manual L2 distance between second and first embedding:", l2_dist_manual[1,0])
print()


l2_dist_scipy = scipy.spatial.distance.cdist(embeddings, embeddings, 'euclidean')
print()
print(f"Scipy L2 distance matrix:")
print(l2_dist_scipy)
print()

print()
print("Verifying that the manual L2 distance matrix is equal to the Scipy L2 distance matrix:")
print(np.allclose(l2_dist_manual, l2_dist_scipy))
print()



