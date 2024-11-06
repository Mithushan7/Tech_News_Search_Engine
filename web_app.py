import json
import numpy as np
import os
import string
from collections import defaultdict
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import math
from collections import Counter

# this function reads the inverted index json file
#this will used to calculate the similarity function for VSM and QLM
def load_inverted_index(file_path='inverted_index.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        inverted_index = json.load(file)
    return inverted_index 


# function to create the log term frequency for term in each document
# function takes in two inputs inverted_index and num_documents
def doc_log_tf(inverted_index, num_documents):
    tf_doc = defaultdict(dict)
        #creating a deafult dict whith integere deault value for missing values
        #reson- to avoid keyerror 
    idf = {}

    #loop to iterate the key word in inverted index and its corresponding documents
    for term, postings in inverted_index.items():
        doc_frequency = len(postings) #counts the documents with that particular term
        #gets the log- inverse document frequency of the words 
        idf[term] = np.log(num_documents / (doc_frequency + 1))

        #outer and inner loop to calculate the log tf of the dcoument 
        #to create a tf document vector 
        #iterates through each unique term, posting which the doc_id and term position
    for term, postings in inverted_index.items():
        for doc_id in postings:
            tf = len(postings[doc_id]) #gets the numerica term frequency
            #if statement to apply log to dampen high frequency words
            if tf > 0:
                log_tf = 1 + np.log(tf)
            else:
                log_tf = 0
            tf_doc[doc_id][term] = log_tf
    
    return tf_doc, idf

# Create document vectors and store the first line
def create_document_vectors(tf_doc, all_terms):
    
        #initializing the data structure to store the document vector and URL
    document_vectors = {}#dictonary to store each documents log term frequency
        #list is being created containing all unique term in the scrapped courpus 
        #creating the index of unique term to create table linke strucutre for log term frequency for documents
        #when creating the document vector we can easily map each term's frequncy based on their position in the list for all documents 
    term_list = list(all_terms)
        #consitency across all documents when creating the document log tf

    
    document_lines = {}
        #dictonary to store first line of each document where keys are doc name
    document_urls = {}
        #dictonary to store URL of each document key is doc name value in corresponding URL

        #loop (for loop) to iterate over each document and the its calcualted log term frequency
        #vector (numpy array) is initialized with the length equal to all term in the above mentioned list ("term_list")
    for doc_id, terms in tf_doc.items():
        vector = np.zeros(len(term_list))

            #inner loop
            #iterates over term_list if the current term in found in the document
            #the document is vector is updated with the corresponding document terms frequency
        for i, term in enumerate(term_list):
            if term in terms:
                vector[i] = terms[term]
        document_vectors[doc_id] = vector

            ##
                 #terms is dictonary containing the log tf of each document
                 # term represents all unique terms in the term_list
                #if a unique term is present in the terms (document) then doc vector is updated
            ##
        
            #loop to retrieve the URL
            #creating a full path to crawled_page folder
        file_path = os.path.join('crawled_pages', doc_id)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                first_line = file.readline().strip() #reads only the first line without white spcae
                document_lines[doc_id] = first_line
                    #stores all first line of the document in a dictonary with doc_id as the key and 

                if first_line.startswith("URL:"):
                    url = first_line.split("URL:",1)[1].strip()
                    document_urls[doc_id]=url
                        #split the URL based on the first oocurence of the mentioned delimiter "URL:"
                        #[1] gets the second elements of the list which the absolute URL
                        #saves the URL in dictonary where corresponding key is the dictonary and value is the URL
                else:
                    document_urls[doc_id] = "No URL found"
        else:
            document_lines[doc_id] = "File not found"
            document_urls[doc_id] = "No URL found"

    return document_vectors, term_list, document_lines, document_urls

####
# QLM Language model for all 400 documents 
def document_language_model(inverted_index, total_terms_collection):
    language_models = {} #initializing empty dictonary
    doc_lengths = defaultdict(int) 
    #default dictonary to encounter type error deafulty assigns 0 for missing values
    #stores the count of the total terms in each document
    

    #iterative nested loop
    #outer loop iterates over each unique term
    #inner loop interates over each doc_id to get the list of term postion
    #len in used to compute the each word count in each document
    for term, postings in inverted_index.items():
        for doc_id, posting in postings.items():
            tf = len(posting)
            doc_lengths[doc_id] += tf
            #doc_length is a dictonary with id being doc_id and value being count of words in the document
    
    for term, postings in inverted_index.items():
        for doc_id, posting in postings.items():
            tf = len(posting)
            if doc_id not in language_models:
                language_models[doc_id] = {}
            language_models[doc_id][term] = tf / doc_lengths[doc_id]

            ##
            #the language model created above is nested dictonary 
            #outer dictonary key is doc_id
            #inner dictonary key is term
            #innter dictonary value is probabilities of term frequency in each document
            #outer dictonary value is inner dictionary
            ##

    
    return language_models, doc_lengths

# QLM linear interporaltion
def linear_interpolation_JM_smoothing(query_terms, language_models, doc_lengths, total_terms_collection, inverted_index, lambda_param=0.7):
    query_likelihoods = {}
    #initializing a emply dictonary "query_likelihoods" where liklihood of each document to the query will be stored
    epsilon = 1e-10
    #epsilon is added to avoid log(0)
    
    # gets the count of entered query terms
    #uses "counter" class which gets the frequncy of occurence for each term sperately
    query_term_freq = Counter(query_terms) #dictonary

    #loops over each document in language model dictonary to calculate the liklihood
    for doc_id, model in language_models.items():
        score = 0.0 
        # gets the total word count in each documents (1 to 400)
        doc_length = doc_lengths[doc_id]

        for term in query_terms:
            #looks up the "query_term_freq" dictonary and gets the count of each individual term in the query
            query_term_count = query_term_freq[term]
            
            # c(w, d) of the term term in current document
            #if statement assigns 0 if the word is not present in the current document
            if term in model:
                doc_term_freq = model [term]
            else:
                doc_term_freq = 0
            
            # p(w | C): to calcualte the probability of the corresponding term in the entire collection
            # "collection_term_freq" numerical variable gets the count of the word in the enitre courpus
            collection_term_freq = 0
            if term in inverted_index:
                for postings in inverted_index[term].values():
                    collection_term_freq += len(postings)
                    
            #calculating the probability of each term in the courpus 
            if total_terms_collection > 0:
                p_w_given_C = collection_term_freq / total_terms_collection
            else:
                p_w_given_C = 0
            
            # application of linear interpolation smoothing (jelinek_mercer smoothing)
            if doc_length > 0 and p_w_given_C > 0:
                term_prob = (1 + (1 - lambda_param) / lambda_param * (doc_term_freq / (doc_length * p_w_given_C)))+epsilon
                log_term_prob = math.log(term_prob)
                
                # Accumulate the score using the query term frequency
                score += query_term_count * log_term_prob

        # Stores the final likelihood score for each term in a dictonary
        query_likelihoods[doc_id] = score

    return query_likelihoods



# Rank documents for VSM and QLM
def cosine_similarity_ranking(query_vector, document_vectors):
    #intializing empty dictonary to store the score of each document vector to the query
    scores = {}

    for doc_id, doc_vector in document_vectors.items():
        dot_product = np.dot(query_vector, doc_vector)

        #np.linalg computes the eculiadean distance norm 
        norm_q = np.linalg.norm(query_vector)
        norm_d = np.linalg.norm(doc_vector)
        
        #computes the cosine similarity 
        if norm_q and norm_d:
            scores[doc_id] = dot_product / (norm_q * norm_d)
        else:
            scores[doc_id] = 0.0
        
        # ###
        # ## Using Bubble sort to sort the VSM scores
        # #converting the dictonary to list with tuple as the inner element with key value pair of doc_id and cosine similarity score
        # score_list = list(scores.items())
        # for i in range(len(score_list)):
        #     for j in range(i + 1, len(score_list)):
        #          if score_list[i][1] < score_list[j][1]:  # Comparing scores
        #                     # tuple swap if the previous elment score is less than the forward element socre
        #              score_list[i], score_list[j] = score_list[j], score_list[i]

        # top_10_scores = score_list[:10]
        # ###
        # ## End of bubble sort

        
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10]

#ranking for QLM model
def QLM_ranking(query_likelihoods):
    return sorted(query_likelihoods.items(), key=lambda x: x[1], reverse=True)[:10]

# Preprocessing and Query vector for VSM
def preprocess_query(query):
    query = query.lower().translate(str.maketrans('', '', string.punctuation))
    #string punctuation is predefined python string containing common punctuation marks
    return query.split()

def compute_query_vector(query, idf, all_terms):
    #earlier function is called upon to get the query term preprocessed
    preprocessed_query_terms = preprocess_query(query)
    query_vector = np.zeros(len(all_terms))
    #create query vector with same length as document vector

    #gets all unique terms
    term_list = list(all_terms)
    
    for term in preprocessed_query_terms:
        if term in idf:
            tf = preprocessed_query_terms.count(term)
            if tf > 0:
                log_tf = 1 + np.log(tf)  # Compute log TF
            else:
                log_tf = 0
            
            # Get the index of the term
            index = term_list.index(term)
            
            # applying TF-IDF to query vector 
            query_vector[index] = log_tf * idf[term]  

    return query_vector


#calling the functions
inverted_index = load_inverted_index()
#calculates the total number of unique documents in the courpus
num_documents =0
document_set = set()
for docs in inverted_index.values():
    for doc in docs:
        document_set.add(doc)
num_documents = len(document_set)

tf_doc, all_terms = doc_log_tf(inverted_index, num_documents)
#the above code is where all term has been initialized 
document_vectors, term_list, document_lines,document_urls = create_document_vectors(tf_doc, all_terms)

#loop to get the total word count in entire courpus all 400 pages
total_terms_collection = 0
for postings in inverted_index.values():
    for posting in postings.values():
        total_terms_collection += len(posting)
        
language_models, doc_lengths = document_language_model(inverted_index, total_terms_collection)





# Initialize the dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Tech News Search", className='text-center mb-4'))]),
    
    dbc.Row([
        dbc.Col([
            dcc.Input(id='query-input', type='text', placeholder='Enter your search query', style={'width': '100%'}),
            dbc.Button("Search", id='search-button', color='primary', className='mt-2 mb-3 w-100',style={'backgroundColor': '#0b0522', 'borderColor': '#0b0522'}),
        ], width=8)
    ], justify='center'),
    
    dbc.Tabs([
        dbc.Tab(label="VSM",tab_id="vsm"),
        dbc.Tab(label="QLM",tab_id="qlm")
    ], id="tabs", active_tab="vsm", className="justify-content-center"),
    
    dbc.Row([dbc.Col(html.Div(id='search-results', className='mt-4'))], justify='center')
], fluid=True, style={'backgroundColor': '#e2e6f9 ', 'height': '100vh'})

#decorator callback
# Callback to handle search and display results for both VSM and QLM
@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks'), Input('tabs', 'active_tab')],
    [State('query-input', 'value')]
)
def update_output(n_clicks, active_tab, query):
    if n_clicks is None or not query:
        return ""

    if active_tab == "vsm":
        tf_doc, idf = doc_log_tf(inverted_index, num_documents)
        query_vector = compute_query_vector(query, idf, term_list)
        ranked_documents = cosine_similarity_ranking(query_vector, document_vectors)
        title = "Vector space model - Top 10 Results:"
    else:
        query_terms = preprocess_query(query)
        query_likelihoods = linear_interpolation_JM_smoothing(query_terms, language_models, doc_lengths, total_terms_collection, inverted_index, lambda_param=0.7)
        ranked_documents = QLM_ranking(query_likelihoods)
        title = "Query likelihood model - Top 10 Results:"

    result_output = [html.H4(title)]
    for doc_id, score in ranked_documents:
        first_line = document_lines.get(doc_id, "First line not available")
        doc_url = document_urls.get(doc_id, "#")
        result_output.append(html.P([
            f"Document: {doc_id}, First Line: {first_line}, Score: {score:.20f}, ",
            html.A("Click here to open the website", href=doc_url, target="_blank")  
        ]))

    return result_output


if __name__ == "__main__":
    app.run_server(debug=True)
