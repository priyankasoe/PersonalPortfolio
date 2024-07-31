import spacy 
from spacy.tokens import Doc
import json
import string
import math 
import random 
from gensim.models import Word2Vec 
import numpy as np 

from numpy.linalg import norm 

nlp = spacy.load("en_core_web_sm")

filtered = [] 
unique = []

dim_vectors = 100


# for entry in data: 
#     doc = nlp(entry["content"])
#     sents = list(doc1.sents)
    

# model = Word2Vec(training_corpus, vector_size=20, min_count=1)
# # vector = model.wv['man']
# # print(vector)
# sims = model.wv.most_similar('family', topn=10) 



# data preprocessing 
# for token in doc:
#     if not token.is_punct and token.is_stop:
#         if token not in unique:
#             unique.append(token)

# TF-IDF

def remove_punctuation(word):
    if len(word) > 0 and word[len(word)-1] in string.punctuation:
        return word[0:len(word)-1]
    if len(word) > 0 and word[0] in string.punctuation:
        return word[1:]
    return word 

''' list data is a list of vectors (2D array)'''
def avg(list_data):
    if (len(list_data) == 0):
        return 0
    sums = [0] * len(list_data)
    for val in list_data:
        for dim in range(len(val)):
            sums[dim] += val[dim]

    return sums / len(list_data)

def is_equal(list_data1, list_data2):
    if len(list_data1) != len(list_data2):
        return False
    equal = True
    for i in len(list_data1):
        if list_data1[i] != list_data2[i]:
            equal = False
    return equal

def cosine_similarity(v1, v2):
    A = np.array(v1)
    B = np.array(v2)
    return np.dot(A,B) / (norm(A) * norm(B))

''' get_tf_idf_values(doc_id) --> returns a list of tuples of form (word, tf-idf val)
    doc_id: int specifying the id of the document from "writing_data.json" 
'''
def get_tf_idf_values(doc_id):

    curr_doc = {} 

    for doc in data:
        if doc["id"] == doc_id:
            curr_doc = doc
            break 

    num_docs = len(data)
    curr_wordbank = curr_doc["content"].split(" "); 

    # remove punctuation from any words
    for i in range(len(curr_wordbank)):
        curr_wordbank[i] = remove_punctuation(curr_wordbank[i])

    unique_wordbank = []  # unique words in specified document 
    for word in curr_wordbank:
        if word not in unique_wordbank:
            unique_wordbank.append(word) 

    # document frequency - how many documents the word appears in 
    # inverse document frequency - N (num documents) / doc_freq
    # tf * idf
    tf_idf = []

    for curr_word in unique_wordbank:
        
        # TF - term frequency
        tf = curr_wordbank.count(curr_word) / len(curr_wordbank)

        # IDF 
        doc_freq = 0

        # document frequency 
        for doc in data:
            doc_words = doc["content"].split(" ") 
            for word in doc_words: 
                if curr_word == remove_punctuation(word):
                    doc_freq += 1
                    break
        
        idf = num_docs / doc_freq

        tf_idf.append((curr_word, tf*idf))

    return tf_idf

        # if tf*idf > 0.1:
        #     rel_words.append((curr_word,tf*idf))
        #     # print("curr word: "+curr_word+", tf-idf: "+str(tf*idf)+"\n")

        # return rel_words


# k means

''' returns k clusters of words as result of k means algorithm 
data: list of tuples (word, vector embedding) 
k: desired clusters
'''
def k_means(data, k, max_iter):

    data = []; 

    # init centroids 
    prev_centroids = [[0] * dim_vectors ] * k
    curr_centroids = [[0] * dim_vectors ] * k
    for i in range(k):
        for j in range(dim_vectors):
            curr_centroids[i][j] = 2*random.random() - 1  # generate random numbers between -1 and 1

    it = 0 

    while (not is_equal(prev_centroids, curr_centroids) or it == max_iter):

        # list of K lists (representing clusters) 
        # each of K lists will contain some N vectors with 1 <= N <= len(data)
        clustered_data = []
        clustered_words = [] 
        for i in range(k):
            clustered_data[i].append([]) 
            clustered_words[i].append([])

        # assigning clusters 
        for x_i in range(len(data)):
            x = data[x_i][1]  # extract embedding 
            min_error = cosine_similarity(x, curr_centroids[0]^2)  # init min error 
            cluster = 0
            for c_i in range(1,k):
                c = curr_centroids[c_i]
                error = cosine_similarity(x, c)
                if error < min_error:
                    min_error = error
                    cluster = c_i
            clustered_data[cluster].append(x)
            clustered_words[cluster].append(data[x_i][0])

        # set prev centroids for comparison 
        prev_centroids = curr_centroids

        # reassigning centroids 
        for c_i in range(k):
            curr_centroids[c_i] = avg(clustered_data[c_i])

        it += 1 

    return clustered_words

f = open('writing_data.json')
data = json.load(f)

# for entry in data: 
#     if entry["id"] == 19:
#         doc1 = nlp(entry["content"])

# sim = doc1.similarity(doc2) 
# print(sim)

training_corpus = []
data = []

for entry in data:
    doc = nlp(entry["content"])
    sents = list(doc.sents)
    for sent in sents:
        sent_word_list = []
        for word in sent.text.split(" "):
            sent_word_list.append(remove_punctuation(word))
        training_corpus.append(sent_word_list) 


model = Word2Vec(training_corpus, size=dim_vectors, min_count=5, sg=1)
vector = model.wv['family']

for sent in training_corpus:
    for word in sent: 
        # TODO: check for word in model because if occurence not high enough, word may not be in model 
        data.append((word, model.wv(word)))

clusters = k_means(data, 5, 100); 
print(clusters) 

# print(vector)
# sims = model.wv.most_similar('family', topn=10) 
# print(len(sims))
# for i in sims:
#     print(i)

# classification 




# load all words

# one hot encoding 

# creatting vector embedding matrix 

# vector --> hidden layer 

# softmax classification 