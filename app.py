import os
import math
from collections import defaultdict
import dash
from dash import dcc, html, Input, Output
import dash_table

# Create the Dash app
app = dash.Dash(__name__)

def load_documents(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents[filename] = file.read()
    return documents

def preprocess_query(query):
    return query.lower().split()

def calculate_document_lengths(documents):
    return {filename: len(content.split()) for filename, content in documents.items()}

def build_word_frequency(documents):
    word_freqs = {}
    for filename, content in documents.items():
        words = content.lower().split()
        freq = defaultdict(int)
        for word in words:
            freq[word] += 1
        word_freqs[filename] = freq
    return word_freqs

def calculate_qlm_score(query, doc_word_freq, doc_length, total_docs, mu):
    score = 0.0
    total_word_count = sum(doc_word_freq.values())
    
    for word in query:
        word_freq = doc_word_freq.get(word, 0)
        prob_word_given_doc = (word_freq + mu * (1 / (total_word_count + mu))) / (doc_length + mu)
        score += math.log(prob_word_given_doc)
    
    return score

def rank_documents(query, documents, doc_lengths, word_freqs, mu=2000):
    ranked_docs = []
    for filename in documents.keys():
        doc_length = doc_lengths[filename]
        score = calculate_qlm_score(query, word_freqs[filename], doc_length, len(documents), mu)
        ranked_docs.append((filename, score))
    
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Normalize scores to be positive
    min_score = ranked_docs[-1][1] if ranked_docs else 0  # Get the least negative score
    normalized_docs = [(filename, score - min_score) for filename, score in ranked_docs]
    
    return normalized_docs

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Query Likelihood Model Search"),
    dcc.Input(id='query-input', type='text', placeholder='Enter your query', style={'width': '400px'}),
    html.Button('Search', id='search-button', n_clicks=0),
    dash_table.DataTable(
        id='results-table',
        columns=[{"name": "Document", "id": "filename"}, {"name": "Score", "id": "score"}],
        data=[],
        style_table={'overflowX': 'auto'},
    )
])

# Callback to update the table based on the input query
@app.callback(
    Output('results-table', 'data'),
    Input('search-button', 'n_clicks'),
    Input('query-input', 'value')
)
def update_results(n_clicks, query):
    if n_clicks > 0 and query:
        input_directory = "filtered_texts"  # Directory with filtered text files
        documents = load_documents(input_directory)
        processed_query = preprocess_query(query)
        doc_lengths = calculate_document_lengths(documents)
        word_freqs = build_word_frequency(documents)
        results = rank_documents(processed_query, documents, doc_lengths, word_freqs)

        return [{"filename": filename, "score": score} for filename, score in results]
    return []

if __name__ == "__main__":
    app.run_server(debug=True)
