import os
import math
from collections import defaultdict

def load_documents(directory):
    """
    Load documents from the specified directory.
    Returns a dictionary where keys are filenames and values are document contents.
    """
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents[filename] = file.read()
    return documents

def preprocess_query(query):
    """
    Preprocess the query by converting it to lowercase and splitting into words.
    """
    return query.lower().split()

def calculate_document_lengths(documents):
    """
    Calculate the length of each document (number of words).
    Returns a dictionary of document lengths.
    """
    doc_lengths = {filename: len(content.split()) for filename, content in documents.items()}
    return doc_lengths

def build_word_frequency(documents):
    """
    Build word frequency for each document.
    Returns a dictionary where keys are document filenames and values are word frequency dictionaries.
    """
    word_freqs = {}
    for filename, content in documents.items():
        words = content.lower().split()
        freq = defaultdict(int)
        for word in words:
            freq[word] += 1
        word_freqs[filename] = freq
    return word_freqs

def calculate_qlm_score(query, doc_word_freq, doc_length, total_docs, mu):
    """
    Calculate the Query Likelihood Model score for a document based on the query.
    """
    score = 0.0
    total_word_count = sum(doc_word_freq.values())
    
    # For each word in the query
    for word in query:
        word_freq = doc_word_freq.get(word, 0)
        prob_word_given_doc = (word_freq + mu * (1 / (total_word_count + mu))) / (doc_length + mu)
        score += math.log(prob_word_given_doc)
    
    return score

def rank_documents(query, documents, doc_lengths, word_freqs, mu=2000):
    """
    Rank documents based on the QLM score for the given query.
    Returns a list of tuples (filename, score) sorted by score.
    """
    ranked_docs = []
    for filename in documents.keys():
        doc_length = doc_lengths[filename]
        score = calculate_qlm_score(query, word_freqs[filename], doc_length, len(documents), mu)
        ranked_docs.append((filename, score))
    
    # Sort by score in descending order
    ranked_docs.sort(key=lambda x: x[1], reverse=True)

    # Normalize scores to be positive
    min_score = ranked_docs[-1][1]  # Get the least negative score
    normalized_docs = [(filename, score - min_score) for filename, score in ranked_docs]
    
    return normalized_docs


def main(input_directory, query, mu=2000):
    # Load documents
    documents = load_documents(input_directory)
    
    # Preprocess the query
    processed_query = preprocess_query(query)

    # Calculate document lengths and word frequencies
    doc_lengths = calculate_document_lengths(documents)
    word_freqs = build_word_frequency(documents)

    # Rank documents based on the query
    ranked_docs = rank_documents(processed_query, documents, doc_lengths, word_freqs, mu)

    # Print the top results
    print("Top relevant documents:")
    for filename, score in ranked_docs[:5]:  # Adjust the slice for more results
        print(f"{filename}: Score = {score:.6f}")

if __name__ == "__main__":
    input_directory = "filtered_texts"  # Directory with filtered text files
    query = "Google Pixel 9 Pro"  # Replace with your actual query
    main(input_directory, query)
