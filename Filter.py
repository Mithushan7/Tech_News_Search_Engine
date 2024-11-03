import os
import re
import json

def extract_json_from_text(content):
    """
    Extract JSON data from the text content using regex to find
    the <script type="application/ld+json"> blocks.
    """
    json_blocks = []
    
    # Regex pattern to extract <script type="application/ld+json"> JSON blocks
    json_pattern = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.DOTALL)
    
    matches = json_pattern.findall(content)
    for match in matches:
        try:
            json_data = json.loads(match)
            json_blocks.append(json_data)
        except json.JSONDecodeError:
            print("Could not decode JSON block. Skipping...")
            continue
    
    return json_blocks

def extract_relevant_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract embedded JSON data
    json_blocks = extract_json_from_text(content)
    
    # Collect relevant text (articleBody, description, etc.)
    text_parts = []
    for block in json_blocks:
        if isinstance(block, dict):
            # Extract 'articleBody' and any other relevant fields
            if 'articleBody' in block:
                text_parts.append(block['articleBody'])
            if 'description' in block:
                text_parts.append(block['description'])

    return "\n\n".join(text_parts) if text_parts else ""

def filter_text_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            relevant_text = extract_relevant_text(file_path)

            if relevant_text:  # Only write if there's relevant text
                output_file_path = os.path.join(output_dir, f"filtered_{filename}")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(relevant_text)
                print(f"Filtered text saved to: {output_file_path}")

if __name__ == "__main__":  # Corrected here from "_main_" to "__main__"
    input_directory = "crawled_pages"  # Your input directory
    output_directory = "filtered_texts"  # Where to save the filtered files
    filter_text_files(input_directory, output_directory)
