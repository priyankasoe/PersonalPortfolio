import spacy 
from spacy.tokens import Doc
import json
import string
import math 
import random 
from gensim.models import Word2Vec 
import numpy as np 

from numpy.linalg import norm 

k = 3 
max_iter = 100


def is_equal(list_data1, list_data2):
    if len(list_data1) != len(list_data2):
        return False
    equal = True
    for i in range(len(list_data1)):
        if list_data1[i] != list_data2[i]:
            equal = False
    return equal

def cosine_similarity(v1, v2):
    A = np.array(v1)
    B = np.array(v2)
    return np.dot(A,B) / (norm(A) * norm(B))


''' list data is a list of tuples 
'''
def avg(list_data):
    if (len(list_data) == 0):
        return 0
    sumY = 0
    sumX = 0
    for val in list_data:
        sumX += val[0]
        sumY += val[1]
    return (sumX / len(list_data), sumY / len(list_data))

'''calculates 2D euclidean distance'''
def euclidean_distance(v1, v2):
    return math.sqrt(math.pow((v1[0] - v2[0]),2) + math.pow((v1[1] - v2[1]), 2))

data = [(1,1), (2,1), (3,3), (3,4), (2,4), (3,5), (4,5), (5,2), (6,1), (6,2), (6,3), (1,2)]; 

min_data = 1
max_data = 6

# init centroids 
prev_centroids = [(0,0)] * k
print(prev_centroids)
curr_centroids = [(0,0)] * k
for i in range(k):
    curr_centroids[i] = (random.randint(min_data, max_data), random.randint(min_data, max_data))
    print(curr_centroids[i])

# list of K lists (representing clusters) 
# each of K lists will contain some N vectors with 1 <= N <= len(data)

it = 0

while (not is_equal(prev_centroids, curr_centroids) or it == max_iter):
    clustered_data = []
    for i in range(k):
        clustered_data.append([])

    # assigning clusters 
    for x_i in range(len(data)):
        x = data[x_i] 
        min_error = euclidean_distance(x, curr_centroids[0])  # init min error 
        nearest_cluster = 0
        for c_i in range(1,k):
            c = curr_centroids[c_i]
            error = euclidean_distance(x, c)
            if error < min_error:
                min_error = error
                nearest_cluster = c_i
        clustered_data[nearest_cluster].append(x)
        # print(clustered_data)
        # print(nearest_cluster)
        # print("---")

    # print("clustered data:")

    # print(clustered_data)
    # print("end")

    # set prev centroids for comparison 
    prev_centroids = curr_centroids

    # reassigning centroids 
    for c_i in range(k):
        curr_centroids[c_i] = avg(clustered_data[c_i])
        print(curr_centroids[c_i])

    it += 1

for i in curr_centroids:
    print(i)