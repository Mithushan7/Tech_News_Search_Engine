import numpy as np
from collections import defaultdict
import string
import json

# Preprocess query: convert to lowercase, remove punctuation, and split into words
def preprocess_query(query):
    query = query.lower().translate(str.maketrans('', '', string.punctuation))
    return query.split()

# Build a basic language model for each document
def build_language_model(inverted_index, total_terms_collection):
    language_models = {}
    
    # Calculate total term frequency in each document
    doc_lengths = defaultdict(int)
    
    for term, postings in inverted_index.items():
        for doc_id, posting in postings.items():
            tf = posting[0]
            doc_lengths[doc_id] += tf  
    
    # Create a probability model for each document
    for term, postings in inverted_index.items():
        for doc_id, posting in postings.items():
            tf = posting[0]  
            if doc_id not in language_models:
                language_models[doc_id] = {}
            # Probability of term in document
            language_models[doc_id][term] = tf / doc_lengths[doc_id]
    
    return language_models, doc_lengths

#query likelihood (Dirichlet smoothing)
def dirichlet_smoothing(query_terms, language_models, doc_lengths, total_terms_collection, inverted_index, mu=2000):
    query_likelihoods = {}
    
    for doc_id, model in language_models.items():
        likelihood = 1.0  
        for term in query_terms:
            doc_term_freq = model.get(term, 0) * doc_lengths[doc_id]  
            collection_term_freq = sum(posting[0] for posting in inverted_index.get(term, {}).values())
            
            # Dirichlet smoothing formula
            term_prob = (doc_term_freq + (mu * (collection_term_freq / total_terms_collection))) / (doc_lengths[doc_id] + mu)
            
            # Multiply the term probability to the likelihood
            likelihood *= term_prob
        
        # Store the final likelihood for the document
        query_likelihoods[doc_id] = likelihood
    
    return query_likelihoods

# Rank documents based on query likelihood scores
def rank_documents_by_query_likelihood(query_likelihoods):
    #sort by likelihood score (top 10)
    return sorted(query_likelihoods.items(), key=lambda x: x[1], reverse=True)[:10]

#loading the inverted index from json
def load_inverted_index(file_path='D:/IR/web crawling/IRA_search_engine/inverted_index.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        inverted_index = json.load(file)
    return inverted_index

# Main function to handle the process
if __name__ == "__main__":
    # Load the inverted index
    inverted_index = load_inverted_index()
    total_terms_collection = sum(len(posting) for term, posting in inverted_index.items()) 

    # Build a simple language model for documents
    language_models, doc_lengths = build_language_model(inverted_index, total_terms_collection)

    # Get a search query from the user
    query = input("Enter your search query: ")
    query_terms = preprocess_query(query)

    # Calculate likelihoods using Dirichlet smoothing
    mu = 5000  
    query_likelihoods = dirichlet_smoothing(query_terms, language_models, doc_lengths, total_terms_collection, inverted_index, mu)

    # Rank documents by query likelihood
    ranked_documents = rank_documents_by_query_likelihood(query_likelihoods)

    #top 10 ranked documents
    print("Top 10 ranked documents:")
    for doc_id, score in ranked_documents:
        print(f"Document ID: {doc_id}, Likelihood: {score:.6f}")

