import json
import numpy as np
from collections import defaultdict
import string

# loading the inverted index from JSON
def load_inverted_index(file_path='inverted_index.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        inverted_index = json.load(file)
    return inverted_index

# calculate TF-IDF scores
def compute_tfidf(inverted_index, num_documents):
    tfidf = defaultdict(dict)
    
    # IDF for each term
    idf = {}
    for term, postings in inverted_index.items():
        doc_frequency = len(postings)  # Number of documents containing the term
        idf[term] = np.log(num_documents / (doc_frequency + 1))  # Smoothing

    # TF-IDF for each document
    for term, postings in inverted_index.items():
        for doc_id in postings:
            tf = postings[doc_id][0]  # Assuming tf is stored as the first element
            tfidf[doc_id][term] = tf * idf[term]
    
    return tfidf, idf.keys()

# create document vectors
def create_document_vectors(tfidf, all_terms):
    document_vectors = {}
    term_list = list(all_terms)  
    
    for doc_id, terms in tfidf.items():
        # Create a zero vector of length equal to the number of unique terms
        vector = np.zeros(len(term_list))
        for i, term in enumerate(term_list):
            if term in terms:
                vector[i] = terms[term]
        document_vectors[doc_id] = vector
    return document_vectors, term_list

# tokenize and preprocess the query
def preprocess_query(query):
    # Convert to lowercase and remove punctuation
    query = query.lower().translate(str.maketrans('', '', string.punctuation))
    return query.split()

# process query and compute its TF-IDF vector
def compute_query_vector(query, inverted_index, idf, all_terms):
    query_terms = preprocess_query(query)
    query_vector = np.zeros(len(all_terms))
    
    term_list = list(all_terms)
    
    for term in query_terms:
        if term in inverted_index:
            term_index = term_list.index(term)
            query_vector[term_index] = 1 * idf[term]  # Simple binary TF
    return query_vector

# compute cosine similarity
def compute_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

# rank documents based on query(top 10 docs)
def rank_documents(query_vector, document_vectors):
    scores = {}
    for doc_id, doc_vector in document_vectors.items():
        scores[doc_id] = compute_cosine_similarity(query_vector, doc_vector)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10]

# Main function
if __name__ == "__main__":
    # Load the inverted index from JSON
    inverted_index = load_inverted_index()
    num_documents = len(set(doc for docs in inverted_index.values() for doc in docs))  

    # TF-IDF for documents
    tfidf, all_terms = compute_tfidf(inverted_index, num_documents)

    # document vectors
    document_vectors, term_list = create_document_vectors(tfidf, all_terms)

    # Get user query
    query = input("Enter your search query: ")

    # IDF for the query
    idf = {term: np.log(num_documents / (len(inverted_index[term]) + 1)) for term in inverted_index}

    # query vector
    query_vector = compute_query_vector(query, inverted_index, idf, term_list)

    # Rank documents based on the query vector
    ranked_documents = rank_documents(query_vector, document_vectors)

    # Display ranked documents
    print("Top 10 ranked documents:")
    for doc_id, score in ranked_documents:
        print(f"{doc_id}: {score:.4f}")
