import os
import re
import json
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

#the import  re(regular expression) is used for finidng words in text with punctuation and symbols


#donwlaoding lexical database for lemmatization
#'wordnet' contains the synsets for lemmatization process,
nltk.download('wordnet')
nltk.download('omw-1.4') #this library same as wordnet but its for multilingual (enables corss lingual support)

#Average perceptron algorithm for POS tagging which has already pretrained using a large corpus 
#what this does is uses the pretrained weights and bias to predict the current words POS
#POS tagging mean tagging a word as noun, verb etc so lemmatizer can accurateyl bring that partiuclar word to base form usign the pos tag 
nltk.download('averaged_perceptron_tagger_eng')


# initializing the lemmatizer this is what going to break down our verge.com crawled web pages text to basic form
lemmatizer = WordNetLemmatizer()

# POS Tagging dunction
def POS_tagging(word):
    #this function helps the lemmatizer in breaking down a word to its basic form 
    #it ensure a noun or verb is correctly identified as noun and verb 
    #so the lemmatizer knows how to break down the word to basic form using the relevant POS tag
    
    #return a tuple for eversingle word with their corresponding POS tag
    tagger = nltk.pos_tag([word])[0][1][0].upper()

    #create a dictonary to map the word with wordnet constant
    tagger_dictonary = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tagger_dictonary.get(tagger, wordnet.NOUN)  # Default to noun if no tag is found

# function to lemmatize crawled page text
def document_term_lemmatizer(text):
    #this function is tokenize the word by removing characters, symbols and punctuation marks
    #check for word boundaries and removes the punktuation mark
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    #process of actual lemmatization of the individual words
    #takes in the word gets the pos tag from the previous function and lemmatize it (brek down to dictionary form)
    normalized_tokens = []
    for token in tokens:
        pos_tag = POS_tagging(token)
        normalized_token = lemmatizer.lemmatize(token, pos_tag)
        normalized_tokens.append(normalized_token)
    
    #returns a list of words
    return normalized_tokens


# inverted index function
def inverted_indexer(folder_path):
    #this function loops through all crawled pages and gets the lemmantized words
    #creates dictonary storing the word, page and the postion of the word in each page

    inverted_index = {}  #dictonary to store the unique words and the document 

    # Loop which loops through all 400 crawled web pages
    for filename in os.listdir(folder_path):
        if filename.startswith("page"): #accept only pages with name "page"
            file_path = os.path.join(folder_path, filename)
            
    
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # skipping first line since it contains the URL
                
                #making the full text into single string
                # when creating the inverted index its easier obtain the position of the word
                content = ' '.join(lines).strip()
                
                #calling our lemmatizer function
                tokens = document_term_lemmatizer(content)
                
                # for loop to loop all the words
                for position, term in enumerate(tokens):
                    # we are dealing with a nested dictionary 
                    # add the word the in the outer dictonary if it is not there
                    if term not in inverted_index:
                        inverted_index[term] = {}
                    
                    # Initializing the inner dictionary to store the page name for all unique words 
                    if filename not in inverted_index[term]:
                        inverted_index[term][filename] = [] #creates a list to store the word's position
                    
                    # Save the position of the word in the file
                    inverted_index[term][filename].append(position)

    return inverted_index

# function to convert the inverted index dictionary to json file
def inverted_index_Dict_to_JSON(inverted_index, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(inverted_index, json_file, indent=4)

# Main function incoporate all def function
def main():
    folder_path = 'crawled_pages'  
    output_file = 'inverted_index.json'  

    # Create the inverted index from the crawled pages
    inverted_index = inverted_indexer(folder_path)
    
    inverted_index_Dict_to_JSON(inverted_index, output_file)
    print(f"Inverted index saved to {output_file}")

# Run the main function when this script is executed
if __name__ == "__main__":
    main()
