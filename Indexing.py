import os
import re
import json
import nltk
from collections import Counter

nltk.download('punkt')  # Ensure NLTK tokenizers are downloaded

# Initialize an empty dictionary for the inverted index
inverted_index = {}

# Function to tokenize and process text for language model
def tokenize(text):
    return nltk.word_tokenize(text.lower())

# Function to process text for building the inverted index (remove punctuation and tokenize)
def process_text(content):
    # Replace all punctuation with spaces using regular expressions
    content = re.sub(r'[^\w\s]', ' ', content)
    # Convert to lowercase and split into tokens (words)
    tokens = content.lower().split()
    return tokens

# Build the inverted index from the text files
def build_inverted_index(documents):
    document_id = 0

    for doc in documents:
        document_id += 1

        # Process the text to extract tokens
        tokens = process_text(doc)

        # Update the inverted index
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = set()  # Use a set to avoid duplicate document IDs
            inverted_index[token].add(document_id)

    # Convert sets to lists for JSON serialization
    for token in inverted_index:
        inverted_index[token] = list(inverted_index[token])

# Function to save the inverted index to a JSON file
def save_inverted_index(output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, indent=4)

# Function to build language models (term frequency) for each document
def build_language_model(documents):
    collection_model = Counter()  # Collect term frequencies across all documents
    document_models = []

    for doc in documents:
        tokenized_doc = tokenize(doc)
        doc_model = Counter(tokenized_doc)
        collection_model.update(tokenized_doc)  # Update collection model with the document's terms
        document_models.append(doc_model)
    
    return document_models, collection_model

# Function to read text files from a directory and return their content as a list
def read_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

if __name__ == "__main__":
    input_directory = "filtered_texts"  # Directory with the filtered text files
    documents = read_documents(input_directory)
    
    # Build the inverted index
    build_inverted_index(documents)
    
    # Save the inverted index to a file
    output_file = "inverted_index.json"
    save_inverted_index(output_file)
    print(f"Inverted index saved to {output_file}")
    
    # Build language models for query likelihood model
    document_models, collection_model = build_language_model(documents)
    print("Language models built for QLM.")
