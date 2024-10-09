from bs4 import BeautifulSoup
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to load and extract text from crawled HTML files
def load_crawled_pages():
    documents = []  
    folder_path = 'crawled_pages'  # Folder where HTML files are saved
    
    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        # Only process files with .html extension
        if filename.endswith('.html'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read the HTML content of the file
                html_content = file.read()
                # Use BeautifulSoup to extract the text from the HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                text = soup.get_text()  # Get the text without HTML tags
                
                # Only append non-empty text to documents list
                if text.strip():
                    documents.append(text)
    
    return documents  # Return the list of documents

# Load the documents from the crawled HTML pages
documents = load_crawled_pages()

# Check if the documents list is empty
if not documents:
    print("No valid documents found.")
else:
    # Step 1: Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Step 2: Fit and transform the documents to generate the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Step 3: Convert the sparse matrix to an array (dense matrix)
    tfidf_array = tfidf_matrix.toarray()

    # Step 4: Get the terms (features) from the vectorizer
    terms = vectorizer.get_feature_names_out()

    # Step 5: Get user input for the query
    query_input = input("Enter your search query: ").strip()  # Input from the user
    query = [query_input]  # Create a list from the input

    # Step 6: Transform the query into the same TF-IDF vector space
    query_vector = vectorizer.transform(query).toarray()

    # Step 7: Compute cosine similarity between the query and all document vectors
    cosine_sim = cosine_similarity(query_vector, tfidf_array)

    # Step 8: Print the similarity scores for each document
    print("Cosine Similarity Scores:")
    print(cosine_sim)

    # Step 9: Rank the documents based on similarity
    # Sort the indexes of cosine similarity in descending order
    ranking = np.argsort(-cosine_sim[0])  # Sort in descending order
    print("\nRanked Document Indexes (from most relevant to least relevant):", ranking)

    # Step 10: Display the most relevant document
    most_relevant_idx = ranking[0]  # Get the index of the most relevant document
    print("\nMost Relevant Document:")
    print(f"Document {most_relevant_idx}:")
    print(documents[most_relevant_idx])  # Print the content of the most relevant document
